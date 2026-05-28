"""Inference helper for the font classifier (ResNet-34 ONNX, INT8 quantized).

Predicts (font_family, is_bold, is_italic) from a cropped RGB image of text.

Design constraints:
- Soft-fail: if the model or onnxruntime is missing, returns None so callers
  can fall back to existing heuristic detection.
- Lazy-load: the ONNX session is built once on first call and reused.
- Thread-safe initialization.
"""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np


_LOCK = threading.Lock()
_SESSION = None
_META: dict[str, Any] | None = None
_MODEL_DIR_DEFAULT = (
    Path(__file__).resolve().parents[3] / "models"
)


def _resolve_paths(model_path: str | Path | None, meta_path: str | Path | None):
    if model_path is None:
        model_path = _MODEL_DIR_DEFAULT / "font_classifier.onnx"
    model_path = Path(model_path)
    if meta_path is None:
        meta_path = model_path.with_suffix(".meta.json")
    return Path(model_path), Path(meta_path)


def _try_load(model_path: str | Path | None = None, meta_path: str | Path | None = None):
    global _SESSION, _META
    if _SESSION is not None:
        return
    model_path, meta_path = _resolve_paths(model_path, meta_path)
    if not model_path.exists() or not meta_path.exists():
        return
    try:
        import onnxruntime as ort
    except ImportError:
        return
    _SESSION = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    _META = json.loads(meta_path.read_text())


def is_available() -> bool:
    """Returns True if the model loaded successfully on first attempt."""
    with _LOCK:
        _try_load()
    return _SESSION is not None and _META is not None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _preprocess(crop: np.ndarray, img_size: int) -> np.ndarray:
    """Scale crop so glyph height = img_size, then either pad to square (short
    crops) or split into overlapping square windows (long crops).

    Long OCR rows (aspect > 3:1) used to get squashed into a 4-pixel band when
    naively rescaled to a square — strokes disappeared and the model fell back
    to spurious classes like Consolas. Sliding windows keep each tile at the
    crop's native glyph scale, so each window is on the same distribution as
    training data.

    Returns a stack of NCHW samples (one row per window, batch dim = N).
    """
    from PIL import Image
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    elif crop.shape[2] == 4:
        crop = crop[..., :3]
    img = Image.fromarray(crop.astype(np.uint8), mode="RGB")
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        raise ValueError(f"Invalid crop size {iw}x{ih}")

    # The training data was synthesized at ~50% glyph-fill: characters occupied
    # roughly half of the img_size canvas, surrounded by background padding. If
    # we naively scale the glyph height to img_size (100% fill), single-window
    # inference sees a distribution it never trained on and confidences collapse.
    # Aim for the glyph-fill ratio used at training time.
    glyph_fill = 0.55
    target_short_side = max(1, int(round(img_size * glyph_fill)))
    scale = target_short_side / min(iw, ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    img_resized = img.resize((nw, nh), Image.BILINEAR)

    corners = [img.getpixel((0, 0)), img.getpixel((iw - 1, 0)),
               img.getpixel((0, ih - 1)), img.getpixel((iw - 1, ih - 1))]
    bg = tuple(int(np.mean([c[k] for c in corners])) for k in range(3))

    windows = []
    if nw <= img_size and nh <= img_size:
        pad = Image.new("RGB", (img_size, img_size), bg)
        pad.paste(img_resized, ((img_size - nw) // 2, (img_size - nh) // 2))
        windows.append(pad)
    else:
        # Slide along the long axis. 50% overlap reduces window-boundary bias.
        stride = max(1, img_size // 2)
        if nw >= nh:
            # Horizontal text row: slide horizontally
            x = 0
            while True:
                end_x = min(x + img_size, nw)
                start_x = max(0, end_x - img_size)
                tile = img_resized.crop((start_x, 0, end_x, nh))
                # If tile is shorter than img_size in either axis, pad.
                pad = Image.new("RGB", (img_size, img_size), bg)
                tw, th = tile.size
                pad.paste(tile, ((img_size - tw) // 2, (img_size - th) // 2))
                windows.append(pad)
                if end_x >= nw:
                    break
                x += stride
        else:
            y = 0
            while True:
                end_y = min(y + img_size, nh)
                start_y = max(0, end_y - img_size)
                tile = img_resized.crop((0, start_y, nw, end_y))
                pad = Image.new("RGB", (img_size, img_size), bg)
                tw, th = tile.size
                pad.paste(tile, ((img_size - tw) // 2, (img_size - th) // 2))
                windows.append(pad)
                if end_y >= nh:
                    break
                y += stride

    mean = np.array(_META["normalization"]["mean"], dtype=np.float32)
    std = np.array(_META["normalization"]["std"], dtype=np.float32)
    batch = []
    for w in windows:
        arr = np.asarray(w, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        batch.append(arr.transpose(2, 0, 1))
    return np.stack(batch, axis=0)


def predict_font(
    crop: np.ndarray,
    *,
    bgr: bool = False,
    model_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Run inference on a HxWx3 uint8 image crop.

    bgr: set True when the crop comes from cv2.imread (BGR channel order).
    Returns None if the model isn't available. Otherwise:
        {
          "family": str,
          "family_confidence": float,
          "is_bold": bool,
          "is_italic": bool,
          "bold_confidence": float,
          "italic_confidence": float,
        }
    """
    with _LOCK:
        _try_load(model_path)
        if _SESSION is None or _META is None:
            return None
    img_size = int(_META.get("img_size", 128))
    if bgr and crop.ndim == 3 and crop.shape[2] >= 3:
        crop = crop[..., ::-1]  # BGR -> RGB
    try:
        x = _preprocess(crop, img_size)
    except Exception:
        return None
    family_logits, bold_logit, italic_logit = _SESSION.run(None, {"image": x})
    # x may contain multiple sliding windows for long crops. The whole row
    # is assumed to share one (family, bold, italic), so we average logits.
    fam_logits_avg = family_logits.mean(axis=0)
    fam_probs = _softmax(fam_logits_avg)
    fam_idx = int(np.argmax(fam_probs))
    fams = _META["families"]
    bold_prob = float(_sigmoid(float(bold_logit.mean())))
    ital_prob = float(_sigmoid(float(italic_logit.mean())))
    return {
        "family": fams[fam_idx],
        "family_idx": fam_idx,
        "family_confidence": float(fam_probs[fam_idx]),
        "is_bold": bold_prob > 0.5,
        "is_italic": ital_prob > 0.5,
        "bold_confidence": bold_prob,
        "italic_confidence": ital_prob,
        "n_windows": int(x.shape[0]),
    }


def predict_font_from_bbox(
    source_image: np.ndarray,
    bbox,
    *,
    bgr: bool = True,
    pad: int = 2,
    min_size: int = 8,
) -> dict[str, Any] | None:
    """Crop the bbox from source_image and run font prediction.

    bgr: defaults to True since most Image2PPT pipelines hand around
    cv2.imread BGR arrays. Set False for an RGB source.
    Returns None on degenerate crops or model unavailability.
    """
    h, w = source_image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return predict_font(source_image[y1:y2, x1:x2], bgr=bgr)
