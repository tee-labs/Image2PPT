"""Font discovery + Pillow loading + per-text rasterised metrics cache."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fontconfig_helper import fontconfig_font_path  # noqa: E402


_FONT_CANDIDATES_CACHE: list[dict] | None = None
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_TEXT_RENDER_CACHE: dict[tuple[str, str, int, bool], dict | None] = {}


def default_ppt_font() -> str:
    """Target PPT font family. macOS QA may substitute it locally, but the
    PPTX preserves the intended family name."""
    return "Microsoft YaHei"


def _font_candidates() -> list[dict]:
    """Local render fonts that also have usable PPT font-family names."""
    global _FONT_CANDIDATES_CACHE
    if _FONT_CANDIDATES_CACHE is not None:
        return _FONT_CANDIDATES_CACHE

    yahei_path = fontconfig_font_path("Microsoft YaHei:style=Regular")
    # Keep the default first. One stable CJK sans candidate avoids
    # line-by-line face flipping inside the same paragraph.
    specs: list[tuple[str, str, float]] = []
    if yahei_path:
        specs.append(("Microsoft YaHei", yahei_path, 0.00))
    specs.extend([
        ("Arial Unicode MS", "/Library/Fonts/Arial Unicode.ttf", 0.00),
        ("Arial Unicode MS",
         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0.02),
        ("Hiragino Sans GB", "/System/Library/Fonts/Hiragino Sans GB.ttc",
         0.06),
    ])
    if not Path("/System/Library/Fonts/Hiragino Sans GB.ttc").exists():
        specs.append(("Heiti TC", "/System/Library/Fonts/STHeiti Medium.ttc",
                      0.06))
    seen_paths: set[str] = set()
    out: list[dict] = []
    for ppt_name, raw_path, penalty in specs:
        path = Path(raw_path)
        if not path.exists() or str(path) in seen_paths:
            continue
        try:
            ImageFont.truetype(str(path), 16)
        except OSError:
            continue
        seen_paths.add(str(path))
        out.append({
            "ppt_name": ppt_name,
            "path": str(path),
            "penalty": float(penalty),
        })
    _FONT_CANDIDATES_CACHE = out
    return out


def _load_font(path: str, pixel_size: int) -> ImageFont.FreeTypeFont | None:
    pixel_size = max(1, int(pixel_size))
    key = (path, pixel_size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, pixel_size)
        except OSError:
            return None
    return _FONT_CACHE[key]


def _render_text_metrics(text: str, font_path: str, size_pt: int,
                         pt_per_px: float, bold: bool) -> dict | None:
    """Render one text line locally and return its ink bbox in source px.

    The layout stores font size in points, but the source image is in
    pixels. ``pt_per_px`` converts PPT points back to source pixels so
    the rendered mask can be compared directly with the OCR crop.
    """
    if not text or "\n" in text or pt_per_px <= 0:
        return None
    pixel_size = max(1, int(round(float(size_pt) / pt_per_px)))
    cache_key = (text, font_path, pixel_size, bool(bold))
    if cache_key in _TEXT_RENDER_CACHE:
        return _TEXT_RENDER_CACHE[cache_key]

    font = _load_font(font_path, pixel_size)
    if font is None:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None

    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    if not bbox:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None

    bx1, by1, bx2, by2 = (int(v) for v in bbox)
    pad = max(4, pixel_size // 3)
    canvas_w = max(2, bx2 - bx1 + pad * 2 + (1 if bold else 0))
    canvas_h = max(2, by2 - by1 + pad * 2)
    origin = (pad - bx1, pad - by1)
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    draw.text(origin, text, font=font, fill=255)
    if bold:
        # Synthetic one-pixel emboldening — approximates how PPT/LO
        # fattens regular CJK fonts when no true bold face is available.
        draw.text((origin[0] + 1, origin[1]), text, font=font, fill=255)

    arr = np.array(img)
    ys, xs = np.where(arr > 16)
    if ys.size == 0 or xs.size == 0:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    x_min = int(xs.min()); x_max = int(xs.max()) + 1
    y_min = int(ys.min()); y_max = int(ys.max()) + 1
    rel_bbox = [
        int(x_min - origin[0]), int(y_min - origin[1]),
        int(x_max - origin[0]), int(y_max - origin[1]),
    ]
    mask_crop = arr[y_min:y_max, x_min:x_max] > 16
    metrics = {
        "pixel_size": pixel_size,
        "ink_bbox": rel_bbox,
        "ink_w": int(x_max - x_min),
        "ink_h": int(y_max - y_min),
        "ink_area": int(mask_crop.sum()),
        "mask": mask_crop,
    }
    _TEXT_RENDER_CACHE[cache_key] = metrics
    return metrics
