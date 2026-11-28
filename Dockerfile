FROM python:3.10-slim

# 微信云托管默认时区，日志时间更直观
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.tencent.com/pypi/simple/ -r requirements.txt

# 服务代码
COPY app.py refine.py ./

# 模型：仓库根目录已存放（名为 pphumanseg.onnx / modnet_fp32_static.onnx），
# 这里改名拷进 /app/weights，与 app.py 里 HERE/weights/<file> 的读取路径对齐
RUN mkdir -p /app/weights
COPY pphumanseg.onnx /app/weights/pphumanseg.onnx
COPY modnet_fp32_static.onnx /app/weights/modnet.onnx

# 默认抠图模型：pphumanseg（6MB，最快最省内存）| modnet（26MB，边缘更细）
ENV PORT=80 \
    MATTING_MODEL=pphumanseg
EXPOSE 80

# 端口先开、模型后台加载（app.py 内部实现），首个请求也不至于立刻超时
CMD ["python", "app.py"]
