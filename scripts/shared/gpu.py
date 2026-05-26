"""Centralized GPU detection for the three inference backends used in
the pipeline: PaddlePaddle (PP-OCRv5 + SLANet_plus), PyTorch via EasyOCR,
and ONNX Runtime (RMBG).

Each backend has its own GPU story and its own way of being asked for a
device, so this module exposes one helper per backend instead of a single
"is GPU available?" boolean.

The detection result is gated by the ``DECKWEAVER_DEVICE`` env var:

  - ``auto`` (default): use GPU on each backend that reports it as
    available at import time.
  - ``gpu``: same as auto today — the request goes through; if a backend
    can't honor it (no CUDA wheel installed, no device visible) it falls
    back to CPU and the backend itself emits the warning.
  - ``cpu``: force CPU on every backend regardless of what's available.

Results are cached per process — the probes import heavy frameworks, and
we never want to repeat that work mid-run.
"""
from __future__ import annotations

import os
from functools import lru_cache


def _mode() -> str:
    m = os.environ.get("DECKWEAVER_DEVICE", "auto").strip().lower()
    return m if m in {"auto", "gpu", "cpu"} else "auto"


@lru_cache(maxsize=1)
def paddle_device() -> str:
    """Return ``"gpu"`` or ``"cpu"`` for ``PaddleOCR(device=...)`` and
    ``paddlex.create_model(device=...)``.

    PaddlePaddle ships separate CPU and GPU wheels. The CPU wheel has no
    CUDA symbols at all, so probing via ``is_compiled_with_cuda()`` is
    safe and cheap on both — it just returns False on the CPU wheel.
    """
    mode = _mode()
    if mode == "cpu":
        return "cpu"
    try:
        import paddle  # type: ignore
        if (paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0):
            return "gpu"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def easyocr_use_gpu() -> bool:
    """EasyOCR is PyTorch under the hood. Only return True on CUDA —
    MPS triggers a per-load warning and the speedup on small per-line
    crops is marginal, so MPS stays opt-out by default."""
    mode = _mode()
    if mode == "cpu":
        return False
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


@lru_cache(maxsize=1)
def onnx_providers() -> list[str]:
    """Preferred-first list of execution providers for
    ``onnxruntime.InferenceSession(..., providers=...)``.

    Order: CUDA → CoreML (macOS) → CPU. We filter against
    ``ort.get_available_providers()`` so unsupported entries never reach
    the session constructor (which would otherwise raise)."""
    mode = _mode()
    providers: list[str] = []
    if mode != "cpu":
        try:
            import onnxruntime as ort  # type: ignore
            available = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
        except Exception:
            pass
    providers.append("CPUExecutionProvider")
    return providers


def describe() -> str:
    """One-line human-readable summary, useful for startup logs."""
    return (f"DECKWEAVER_DEVICE={_mode()} "
            f"paddle={paddle_device()} "
            f"easyocr_gpu={easyocr_use_gpu()} "
            f"onnx_providers={','.join(onnx_providers())}")
