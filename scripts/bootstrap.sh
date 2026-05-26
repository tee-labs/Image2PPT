#!/usr/bin/env bash
# One-shot setup for Image2PPT.
#
# What this does:
#   1. Sanity-check Python and pip.
#   2. Install all Python dependencies into the active environment.
#   3. (macOS) Install LibreOffice + Poppler + Tesseract via Homebrew if missing.
#      (Linux) Install via apt if available and not already present.
#   4. Pre-download every model the skill uses (PaddleOCR PP-OCRv5,
#      RMBG-1.4) by calling scripts/warmup.py.
#
# Idempotent: re-running is safe; everything is skipped or cached.
#
# Usage:
#   bash scripts/bootstrap.sh                # auto: GPU wheels if nvidia-smi
#                                            #   is present, otherwise CPU wheels
#   bash scripts/bootstrap.sh --cpu          # force CPU wheels even if a GPU
#                                            #   is detected
#   bash scripts/bootstrap.sh --skip-rmbg    # skip optional RMBG model
#   bash scripts/bootstrap.sh --no-system    # skip system tools
#                                            #   (only pip + warmup)
set -euo pipefail

SKIP_RMBG=0
NO_SYSTEM=0
FORCE_CPU=0
for arg in "$@"; do
    case "$arg" in
        --skip-rmbg)  SKIP_RMBG=1 ;;
        --no-system)  NO_SYSTEM=1 ;;
        --cpu)        FORCE_CPU=1 ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0 ;;
        *)
            echo "Unknown flag: $arg" >&2
            exit 2 ;;
    esac
done

# Auto-detect GPU: install CUDA wheels iff nvidia-smi is on PATH AND it
# actually reports a device. `nvidia-smi -L` exits non-zero (and prints
# nothing) when no driver / device is visible, so this avoids installing
# GPU wheels on hosts that just have the binary lying around.
USE_GPU=0
if [ "$FORCE_CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 \
   && nvidia-smi -L >/dev/null 2>&1; then
    USE_GPU=1
fi

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

echo "=== 1/4 Python sanity check ==="
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH." >&2
    exit 1
fi
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "python3 = $(which python3) (version $PYV)"
if ! python3 -c 'import pip' >/dev/null 2>&1; then
    echo "ERROR: pip not available for this python3." >&2
    exit 1
fi

echo
echo "=== 2/4 Python dependencies ==="
python3 -m pip install --upgrade pip
# Common deps — same regardless of CPU/GPU.
python3 -m pip install \
    python-pptx pillow numpy opencv-python \
    'paddleocr>=3' 'paddlex[ocr]' \
    easyocr pytesseract \
    huggingface_hub

if [ "$USE_GPU" -eq 1 ]; then
    echo
    echo "  NVIDIA GPU detected — installing CUDA wheels"
    echo "  (paddlepaddle-gpu + onnxruntime-gpu). Pass --cpu to opt out."
    # PaddlePaddle ships separate CPU and GPU wheels. Uninstall the CPU
    # wheel first so pip doesn't keep both around with conflicting
    # binaries (which silently picks the wrong one at import time).
    python3 -m pip uninstall -y paddlepaddle onnxruntime || true
    python3 -m pip install paddlepaddle-gpu onnxruntime-gpu
else
    if [ "$FORCE_CPU" -eq 1 ]; then
        echo "  --cpu set: installing CPU wheels."
    else
        echo "  No NVIDIA GPU detected — installing CPU wheels."
    fi
    python3 -m pip install onnxruntime
fi

if [ "$NO_SYSTEM" -eq 0 ]; then
    echo
    echo "=== 3/4 System tools (libreoffice + poppler + tesseract) ==="
    OS="$(uname -s)"
    if [ "$OS" = "Darwin" ]; then
        if ! command -v brew >/dev/null 2>&1; then
            echo "WARN: Homebrew not installed; skipping libreoffice + poppler."
            echo "      Install brew from https://brew.sh and rerun, or pass --no-system."
        else
            command -v soffice >/dev/null 2>&1 || \
                [ -x /Applications/LibreOffice.app/Contents/MacOS/soffice ] || \
                brew install --cask libreoffice
            command -v pdftoppm >/dev/null 2>&1 || brew install poppler
            command -v tesseract >/dev/null 2>&1 || brew install tesseract
            if command -v tesseract >/dev/null 2>&1 && \
               ! tesseract --list-langs 2>/dev/null | grep -qx chi_sim; then
                brew install tesseract-lang
            fi
        fi
    elif [ "$OS" = "Linux" ]; then
        if command -v apt-get >/dev/null 2>&1; then
            need_libre=0
            need_pdf=0
            need_tess=0
            need_tess_lang=0
            command -v soffice  >/dev/null 2>&1 || need_libre=1
            command -v pdftoppm >/dev/null 2>&1 || need_pdf=1
            command -v tesseract >/dev/null 2>&1 || need_tess=1
            if command -v tesseract >/dev/null 2>&1 && \
               ! tesseract --list-langs 2>/dev/null | grep -qx chi_sim; then
                need_tess_lang=1
            fi
            if [ "$need_libre" -eq 1 ] || [ "$need_pdf" -eq 1 ] || \
               [ "$need_tess" -eq 1 ] || [ "$need_tess_lang" -eq 1 ]; then
                pkgs=""
                [ "$need_libre"     -eq 1 ] && pkgs="$pkgs libreoffice"
                [ "$need_pdf"       -eq 1 ] && pkgs="$pkgs poppler-utils"
                [ "$need_tess"      -eq 1 ] && pkgs="$pkgs tesseract-ocr"
                [ "$need_tess_lang" -eq 1 ] && pkgs="$pkgs tesseract-ocr-chi-sim"
                echo "Running: sudo apt-get install$pkgs"
                sudo apt-get update
                # shellcheck disable=SC2086
                sudo apt-get install -y $pkgs
            else
                echo "libreoffice + poppler + tesseract already installed."
            fi
        else
            echo "WARN: apt-get not found; install libreoffice, poppler, and tesseract manually."
        fi
    else
        echo "WARN: unknown OS '$OS'; install libreoffice, poppler, and tesseract manually."
    fi
else
    echo "Skipping system tools (--no-system)."
fi

echo
echo "=== 4/4 Warm up model caches ==="
WARMUP_ARGS=()
[ "$SKIP_RMBG" -eq 1 ] && WARMUP_ARGS+=("--skip-rmbg")
python3 scripts/warmup.py "${WARMUP_ARGS[@]}"

echo
echo "Bootstrap complete. Try a single-page run:"
echo "  python3 scripts/ocr/ocr_paddle.py path/to/slide.jpg > /tmp/ocr.json"
