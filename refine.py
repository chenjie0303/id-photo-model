#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""alpha 边缘精修（纯 numpy + Pillow，不依赖 opencv —— 云托管镜像更小）。

1. 1px 腐蚀：MinFilter(3)，消除模型边缘 halo / 白边
2. 0.8px 高斯羽化：边缘过渡自然，避免锯齿
"""
import numpy as np
from PIL import Image, ImageFilter


def refine_alpha(rgb, mask, erode=1, blur=0.8):
    """rgb: 原图 HxWx3 ndarray（保留接口，便于将来做去溢色）；mask: 0-255 float ndarray。"""
    a = Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8), "L")
    if erode > 0:
        # 3x3 最小值 = 1px 腐蚀
        a = a.filter(ImageFilter.MinFilter(3))
    if blur > 0:
        a = a.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(a).astype(np.float32)
