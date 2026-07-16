# syntax=docker/dockerfile:1
#
# Image2PPT (DeckWeaver) — Web 前后端一体镜像。
#
# 生产形态与 web/start-prod.sh 一致：前端编译进 web/frontend/dist/，
# 由 FastAPI (web.backend.app.main:app) 在单端口 8000 同源提供 SPA + API。
#
# 构建时预下载 PaddleOCR / RMBG 模型（开箱即用、离线可用）。
# 不装 easyocr/torch（交叉验证默认关闭，见 DECKWEAVER_CROSS_VERIFY）。
# 仅 linux/amd64 —— PaddlePaddle 等只发布 amd64 预编译 wheel。
#
# 安全：镜像不内置任何默认口令 / JWT 密钥。启动护栏
# (web/backend/app/main.py::_check_secrets) 会拒绝以弱口令启动，
# 因此运行时必须注入 DECKWEAVER_ADMIN_PASSWORD 与 DECKWEAVER_JWT_SECRET。

# ───────────────────────────── 1. 前端构建 ─────────────────────────────
FROM node:20-bookworm-slim AS frontend

WORKDIR /build

# 先拷依赖清单，利用层缓存
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci

# 再拷源码并构建（产物落在 /build/dist）
COPY web/frontend/ ./
RUN npm run build

# ───────────────────────────── 2. 运行时 ───────────────────────────────
FROM python:3.12-slim AS runtime

# 角色：镜像内以 root 运行（LibreOffice/字体/模型均按系统级安装）。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # ---- 容器内语义合理的默认值（均可在 docker run -e 覆盖） ----
    # 容器内没有 bubblewrap/firejail，关闭文件系统沙箱（rlimit 仍生效）。
    DECKWEAVER_SANDBOX_MODE=none \
    # 镜像由 CI 重建来更新，禁止容器内 git 自更新（避免上游 RCE 面）。
    DECKWEAVER_AUTO_UPDATE=false \
    # 未安装 easyocr，关闭交叉验证。
    DECKWEAVER_CROSS_VERIFY=false \
    DECKWEAVER_PYTHON_BIN=python3 \
    # 首次仍可能按需下载模型 / 字体探测，保留出网。
    DECKWEAVER_SANDBOX_ALLOW_NETWORK=true

# ---- 系统依赖：转换链 + CJK 字体 + opencv 运行库 ----
# LibreOffice/Poppler/Tesseract 用于预览渲染与 PDF 栅格化；
# CJK 字体 + fontconfig 别名保证 Microsoft YaHei 度量正确（见 web/README.md）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        fonts-wqy-microhei \
        fonts-wqy-zenhei \
        fonts-noto-cjk-extra \
        libgl1 \
        libglib2.0-0 \
        # bubblewrap：可让转换子进程的 sandbox=auto 在容器内也能生效（可选）。
        bubblewrap \
    && rm -rf /var/lib/apt/lists/*

# fontconfig 别名：把 Microsoft YaHei / 微软雅黑 / PingFang SC
# 映射到与 YaHei 度量兼容的开源字体 WenQuanYi Micro Hei / Noto Sans CJK SC。
RUN mkdir -p /etc/fonts/conf.d && printf '%s\n' \
    '<?xml version="1.0"?>' \
    '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">' \
    '<fontconfig>' \
    '  <alias binding="strong">' \
    '    <family>Microsoft YaHei</family>' \
    '    <prefer><family>WenQuanYi Micro Hei</family><family>Noto Sans CJK SC</family></prefer>' \
    '  </alias>' \
    '  <alias binding="strong">' \
    '    <family>微软雅黑</family>' \
    '    <prefer><family>WenQuanYi Micro Hei</family></prefer>' \
    '  </alias>' \
    '  <alias binding="strong">' \
    '    <family>PingFang SC</family>' \
    '    <prefer><family>WenQuanYi Micro Hei</family><family>Noto Sans CJK SC</family></prefer>' \
    '  </alias>' \
    '</fontconfig>' > /etc/fonts/conf.d/30-yahei.conf \
    && fc-cache -f

WORKDIR /app

# ---- Python 依赖：先装（利用层缓存） ----
# web 层依赖。注意 COPY 进当前目录后用对应路径引用（避免路径不一致）。
COPY web/backend/requirements.txt /tmp/web-requirements.txt
RUN pip install -r /tmp/web-requirements.txt

# CLI 流水线依赖。镜像内不装 easyocr（会拉取 torch ~1GB，交叉验证默认关闭）。
# 列表与根 requirements.txt 保持一致，剔除 easyocr；如后者变更请同步这里。
# 额外显式装 paddlepaddle（CPU 轮）：PaddleOCR 3.x 不硬依赖引擎，
# 缺它时 warmup 报 "Engine 'paddle_static' is unavailable"。CPU 镜像选 paddlepaddle
# （GPU 见 bootstrap.sh 的 paddlepaddle-gpu 分支）。
RUN pip install \
        python-pptx \
        pillow \
        numpy \
        opencv-python \
        'paddleocr>=3' \
        'paddlex[ocr]' \
        paddlepaddle \
        pytesseract \
        onnxruntime \
        huggingface_hub \
        pymupdf

# ---- 拷贝仓库主体（代码 + scripts + 已打包的 models/font_classifier.onnx） ----
COPY . /app

# ---- 构建期预下载模型（PaddleOCR PP-OCRv5 + RMBG-1.4） ----
# warmup.py 仅依赖 paddleocr / huggingface_hub / onnxruntime，不含 easyocr/torch。
RUN python3 scripts/warmup.py

# ---- 从前端阶段拷入已构建的静态产物 ----
COPY --from=frontend /build/dist /app/web/frontend/dist

# 运行时状态（SQLite + 上传 + 产物）全部落在 web/data/，建议挂卷持久化。
VOLUME ["/app/web/data"]

EXPOSE 8000

# 与 web/start-prod.sh 等价：单 uvicorn 进程同源提供 API + SPA。
CMD ["python3", "-m", "uvicorn", "web.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
