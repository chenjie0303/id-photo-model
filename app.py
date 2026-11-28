#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""微信云托管 · 人像抠图服务（多模型可切换）

支持模型（均在 0.5GB 规格内可跑）：
    pphumanseg  PP-HumanSeg 6MB   输入 192  推理 ~20ms  峰值 88MB   最快最省
    modnet      MODNet 26MB       输入 512  推理 ~1s    峰值 320MB  边缘更细

切换方式（二选一，都不需要改代码）：
    1) 环境变量 MATTING_MODEL=modnet       （部署时定默认值）
    2) 请求体 {"model": "modnet"}          （运行时逐个请求切换）

内存策略：只驻留最近使用的一个模型。请求另一个模型时会释放旧的再加载，
峰值始终等于单个模型，0.5GB 规格安全。代价是切换后首请求多 0.3-0.8s 加载。

职责单一：生活照/复杂背景 → 透明 PNG 人像。
换底、裁切、美颜、排版仍在小程序本机完成，服务端不存图。

接口：
    GET  /health          -> {"status": true, "ready": bool, "models": [...], "default": "..."}
    POST /matting         {"image": "<base64>", "model": "pphumanseg|modnet", "fine": true}
         -> {"status": true, "image": "<base64 rgba png>", "width":.., "height":.., "ms":..}

隐私：图片只在内存处理，不落盘、不打日志。
"""
import base64
import gc
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from refine import refine_alpha

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "80"))
MAX_INPUT_BYTES = 12 * 1024 * 1024   # 原图上限 12MB
MAX_SIDE = 2000                       # 超大图先缩到长边 2000，省内存防超时

# ---------- 防刷限流（按 openid，进程内计数，不存用户数据）----------
# 小程序通过 wx.cloud.callContainer 调内网链路时，云托管会自动注入 X-WX-OPENID，
# 服务端据此限流。目的是防止脚本批量刷算力把账单刷爆，不是运营门槛。
# 正常用户一张照片最多重试两三次，6 次/分钟、120 次/天绰绰有余。
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "6"))
RATE_PER_DAY = int(os.environ.get("RATE_PER_DAY", "120"))
RL_MAX_KEYS = 20000
_rl = {}
_rl_lock = threading.Lock()


def rate_allow(key):
    """整数窗口计数（1 分钟 / 1 天）。返回 (是否放行, 建议等待秒数)。

    key 为空（公网直连、本地调试）不限流，靠控制台「最大副本数」兜底。
    多副本时各实例独立计数，实际阈值约为 阈值 × 副本数，属于可接受的宽松限流。
    """
    if not key:
        return True, 0
    now = time.time()
    minute = int(now // 60)
    day = int(now // 86400)
    with _rl_lock:
        st = _rl.get(key)
        if st is None or st[0] != day:
            st = [day, 0, minute, 0]
            _rl[key] = st
        if st[2] != minute:
            st[2] = minute
            st[3] = 0
        if st[1] >= RATE_PER_DAY:
            return False, 120
        if st[3] >= RATE_PER_MIN:
            return False, 60 - int(now % 60)
        st[1] += 1
        st[3] += 1
        # 防止字典无限膨胀：超量时先清掉非今日记录，仍然超量就整体清零
        if len(_rl) > RL_MAX_KEYS:
            stale = [k for k, v in _rl.items() if v[0] != day]
            for k in stale:
                _rl.pop(k, None)
            if len(_rl) > RL_MAX_KEYS:
                _rl.clear()
    return True, 0

# 模型注册表：新增模型只需在这里加一行 + 放权重文件
MODELS = {
    "pphumanseg": {
        "file": "pphumanseg.onnx",
        "size": 192,
        "post": "softmax2",     # (1,2,S,S) logits -> softmax 取前景通道
        "blur": 0.8,            # 羽化半径（PP 输出偏平滑，留一点羽化去锯齿）
    },
    "modnet": {
        "file": "modnet.onnx",
        "size": 512,
        "post": "alpha",        # (1,1,S,S) 已是 0~1 alpha，直接拉伸
        "blur": 0.4,            # MODNet 输出自带软过渡，羽化要更轻，避免发虚
    },
}
# 缺权重的模型直接剔除：只带了部分模型（或构建时下载失败）时，服务仍能起来，
# 不会因为一个文件缺失导致整个服务 500
_missing = [k for k, v in MODELS.items() if not os.path.exists(os.path.join(HERE, "weights", v["file"]))]
if _missing and len(_missing) < len(MODELS):
    for k in _missing:
        MODELS.pop(k, None)
    print("skip models without weights:", _missing, flush=True)

DEFAULT_MODEL = os.environ.get("MATTING_MODEL", "pphumanseg")
if DEFAULT_MODEL not in MODELS:
    DEFAULT_MODEL = "pphumanseg"

_sessions = {}          # name -> (session, input_name)
_lock = threading.Lock()


def get_session(name):
    """取模型会话；内存里只保留一个，切模型时释放旧的。"""
    with _lock:
        if name in _sessions:
            return _sessions[name]
        # 卸载其他模型，保证峰值内存 = 单个模型
        if _sessions:
            _sessions.clear()
            gc.collect()
        # onnxruntime 体积大、import 慢，放到这里按需加载：进程一起来端口就能开，
        # 冷启动时 /health 立刻可用，模型在后台线程里加载
        import onnxruntime as ort
        cfg = MODELS[name]
        path = os.path.join(HERE, "weights", cfg["file"])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 限制线程数，避免小规格实例上多线程争抢反而变慢
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        _sessions[name] = (sess, sess.get_inputs()[0].name)
        return _sessions[name]


def _infer_alpha(name, rgb_arr):
    """对给定 RGB 数组推理一次，返回与原数组同尺寸的 0~1 alpha。

    直接拉伸到模型输入尺寸：人像充满画面，模型精度最高。
    裁剪精修阶段传入的已是正方形区域，拉伸不产生变形。
    """
    cfg = MODELS[name]
    S = cfg["size"]
    sess, in_name = get_session(name)
    h, w = rgb_arr.shape[:2]

    im = Image.fromarray(rgb_arr).resize((S, S), Image.BILINEAR)
    x = np.asarray(im).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).astype(np.float32).transpose(2, 0, 1)[None]

    out = sess.run(None, {in_name: x})[0][0]

    if cfg["post"] == "softmax2":
        e = np.exp(out - out.max(axis=0, keepdims=True))
        p = e / e.sum(axis=0, keepdims=True)
        a = p[1]
        # 该导出模型 logits 幅度很小，必须 min-max 拉伸才有对比度
        a = (a - a.min()) / max(1e-6, float(a.max() - a.min()))
    else:
        # MODNet 输出已是 0~1 alpha。做固定线性拉伸（不用自适应 min-max，
        # 避免单个噪点拉坏整图对比度），再裁剪到合法范围。
        a = np.squeeze(out)
        a = np.clip((a - 0.02) / 0.96, 0.0, 1.0)

    small = Image.fromarray((a * 255).astype(np.uint8), "L")
    return np.asarray(small.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0


def _crop_refine(name, rgb_arr, a1, pad=0.10):
    """两阶段精修：用粗 alpha 定位人像框，裁出来放大到模型输入尺寸再跑一次。

    等效分辨率提升 1.5~2.5 倍，头顶碎发/发丝轮廓明显更清晰，成本仅多一次推理。
    """
    h, w = rgb_arr.shape[:2]
    ys, xs = np.where(a1 > 0.5)
    if len(ys) < 200:
        return a1
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    # 扩成正方形（保持比例不拉伸），再外扩 pad
    side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad))
    if side < 32:
        return a1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    X0, X1 = cx - side // 2, cx + side // 2
    Y0, Y1 = cy - side // 2, cy + side // 2
    # 超出画面时整体平移回画面内，保持正方形
    if X0 < 0:
        X1 -= X0; X0 = 0
    if Y0 < 0:
        Y1 -= Y0; Y0 = 0
    X1, Y1 = min(w, X1), min(h, Y1)
    if X1 - X0 < 32 or Y1 - Y0 < 32:
        return a1
    # 裁出来已接近全图，再跑一次没意义，跳过省算力
    if (X1 - X0) * (Y1 - Y0) > 0.80 * w * h:
        return a1

    sub = rgb_arr[Y0:Y1, X0:X1]
    a2 = _infer_alpha(name, sub)

    out = np.zeros_like(a1)
    out[Y0:Y1, X0:X1] = a2
    # 接缝渐变融合，避免裁剪框处出现硬边
    F = min(8, max(2, (X1 - X0) // 40), max(2, (Y1 - Y0) // 40))
    if Y0 > 0:
        r = np.linspace(0.0, 1.0, F, dtype=np.float32)[:, None]
        out[Y0:Y0 + F, X0:X1] = out[Y0:Y0 + F, X0:X1] * r + a1[Y0:Y0 + F, X0:X1] * (1 - r)
    if Y1 < h:
        r = np.linspace(1.0, 0.0, F, dtype=np.float32)[:, None]
        out[Y1 - F:Y1, X0:X1] = out[Y1 - F:Y1, X0:X1] * r + a1[Y1 - F:Y1, X0:X1] * (1 - r)
    if X0 > 0:
        c = np.linspace(0.0, 1.0, F, dtype=np.float32)[None, :]
        out[Y0:Y1, X0:X0 + F] = out[Y0:Y1, X0:X0 + F] * c + a1[Y0:Y1, X0:X0 + F] * (1 - c)
    if X1 < w:
        c = np.linspace(1.0, 0.0, F, dtype=np.float32)[None, :]
        out[Y0:Y1, X1 - F:X1] = out[Y0:Y1, X1 - F:X1] * c + a1[Y0:Y1, X1 - F:X1] * (1 - c)
    return out


def matting(pil_img, model=None, fine=True, mask_only=False):
    """输入 PIL 图，输出带 alpha 的 RGBA 图。

    model 为空则用 DEFAULT_MODEL；fine=True 启用两阶段裁剪精修。
    mask_only=True 时输出「纯黑 + 透明度」的蒙版：小程序用 destination-in 把原图像素
    套进这个蒙版合成，清晰度不受上传压缩图限制，响应也小得多。
    """
    name = model if model in MODELS else DEFAULT_MODEL
    cfg = MODELS[name]

    orig = pil_img.convert("RGB")
    w, h = orig.size

    # 超大图降采样，避免云托管内存/超时问题
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        orig = orig.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        w, h = orig.size

    rgb = np.asarray(orig)
    a = _infer_alpha(name, rgb)
    if fine:
        a = _crop_refine(name, rgb, a)

    mask = refine_alpha(rgb, a * 255.0, blur=cfg["blur"])
    alpha = Image.fromarray(mask.astype(np.uint8), "L")
    if mask_only:
        # RGB 恒为黑、只有透明度在变：PNG 压得快，客户端也不用逐像素处理
        out = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        out.putalpha(alpha)
        return out

    rgba = orig.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health") or self.path == "/":
            self._json(200, {
                "status": True,
                "ready": bool(_sessions),   # 模型是否已在后台加载完成
                "models": [
                    {"name": k, "size": v["size"], "file": v["file"]}
                    for k, v in MODELS.items()
                ],
                "default": DEFAULT_MODEL,
                "modes": ["image", "mask"],
                "current": list(_sessions.keys())[0] if _sessions else None,
                "rateLimit": {"perMin": RATE_PER_MIN, "perDay": RATE_PER_DAY},
            })
        else:
            self._json(404, {"status": False, "error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/matting"):
            self._json(404, {"status": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_INPUT_BYTES + 2 * 1024 * 1024:
                self._json(413, {"status": False, "error": "请求过大"})
                return

            # 限流在解析图片之前做，被刷时连解码的算力都不给
            openid = self.headers.get("X-WX-OPENID") or ""
            ok, retry = rate_allow(openid)
            if not ok:
                self._json(429, {
                    "status": False,
                    "code": "rate_limited",
                    "error": "生成太频繁，请稍后再试",
                    "retry_after": retry,
                })
                return

            data = json.loads(self.rfile.read(n).decode("utf-8"))
            raw = base64.b64decode(data.get("image", ""))
            if not raw or len(raw) > MAX_INPUT_BYTES:
                self._json(413, {"status": False, "error": "图片过大或为空"})
                return

            model = data.get("model")
            if model and model not in MODELS:
                self._json(400, {"status": False, "error": "不支持的模型: %s" % model})
                return
            name = model if model in MODELS else DEFAULT_MODEL
            fine = bool(data.get("fine", True))
            mask_only = data.get("mode") == "mask"

            # 切模型有加载成本，单独计时方便排查
            need_load = name not in _sessions
            t_load = time.time()
            get_session(name)
            load_ms = int((time.time() - t_load) * 1000) if need_load else 0

            t0 = time.time()
            src = Image.open(io.BytesIO(raw))
            out_img = matting(src, model=name, fine=fine, mask_only=mask_only)
            buf = io.BytesIO()
            # 不用 optimize=True：实测 760ms 只换来 44KB，compress_level=6 仅 157ms。
            # 省下的 0.6 秒全部变成用户少等的时间。
            out_img.save(buf, format="PNG", compress_level=6)
            self._json(200, {
                "status": True,
                "model": name,
                "fine": fine,
                "mode": "mask" if mask_only else "image",
                "image": base64.b64encode(buf.getvalue()).decode("ascii"),
                "width": out_img.width,
                "height": out_img.height,
                "ms": int((time.time() - t0) * 1000),
                "load_ms": load_ms,
            })
        except Exception as e:
            self._json(500, {"status": False, "error": str(e)[:200]})

    def log_message(self, fmt, *args):
        # 不打访问日志，避免泄露用户信息
        pass


def main():
    print("available models:", list(MODELS.keys()), flush=True)
    print("default model:", DEFAULT_MODEL, flush=True)
    # 端口先开、模型后台加载：冷启动后 /health 秒回，小程序启动时的预热能真正提前把模型拉起来
    threading.Thread(target=get_session, args=(DEFAULT_MODEL,), daemon=True).start()
    print("listening on 0.0.0.0:%d (model loading in background)" % PORT, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
