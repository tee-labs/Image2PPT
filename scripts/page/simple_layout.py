#!/usr/bin/env python
"""Simple per-page layout generator: full-page background + OCR text overlays.

Used by the "text-only" build_deck mode. Skips the full icon /
inventory extraction stack and instead emits a minimal layout JSON:

    1) One background image element (the text-erased clean.png) sized
       to the full source canvas. Renders first → behind everything.
    2) One text element per OCR item, with bbox copied from OCR and a
       font size derived directly from bbox height. Renders on top.

The generated JSON uses the same schema as inventory_to_layout.py so
combine_layouts → build_pptx_from_layout consumes it without changes.

Font sizes are best-effort (geometric mapping from pixels to points).
The full pipeline's `calibrate_text_sizes` / `calibrate_text_positions`
stages would normally refine these but are skipped in simple mode —
users wanting tighter visual fidelity should run the full pipeline
instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# 4:3 PPT default. inventory_to_layout uses 7.5" height and derives
# width from source aspect ratio; we mirror that so combined.layout.json
# stays consistent across modes.
DEFAULT_SLIDE_HEIGHT_IN = 7.5

# Font factor: maps OCR bbox height (pixels) to point size. CJK glyphs
# fill the em-box so factor ≈ 1.0 reads correct; Latin lines have
# ascender+descender < em so a slightly smaller factor avoids overflow.
# 0.85 splits the difference for mixed-language slides.
FONT_PT_FACTOR_DEFAULT = 0.85
FONT_PT_FACTOR_LATIN = 0.78

FONT_PT_MIN = 6
FONT_PT_MAX = 96

_CJK_RE = re.compile(
    r"[　-〿぀-ヿ㐀-䶿一-鿿"
    r"豈-﫿＀-￯]"
)


def _is_cjk_heavy(text: str) -> bool:
    if not text:
        return False
    cjk = len(_CJK_RE.findall(text))
    return cjk * 2 >= len(text)


def _font_pt(bbox_h_px: float, source_h_px: int,
             slide_h_in: float, text: str) -> int:
    pt_per_px = (slide_h_in * 72.0) / max(1, source_h_px)
    factor = FONT_PT_FACTOR_DEFAULT if _is_cjk_heavy(text) else FONT_PT_FACTOR_LATIN
    pt = bbox_h_px * pt_per_px * factor
    return int(max(FONT_PT_MIN, min(FONT_PT_MAX, round(pt))))


def _ocr_items(ocr: Any) -> list[dict]:
    """Normalize the OCR JSON to a flat list of {text,x1,y1,x2,y2} dicts.

    Some OCR variants wrap items under "items"/"results" keys; the
    canonical export from this project's prepare_ocr is a top-level
    list. Handle both so simple mode also works on legacy work-dirs.
    """
    if isinstance(ocr, list):
        items = ocr
    elif isinstance(ocr, dict):
        items = ocr.get("items") or ocr.get("results") or ocr.get("ocr") or []
    else:
        items = []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        try:
            x1 = int(it["x1"]); y1 = int(it["y1"])
            x2 = int(it["x2"]); y2 = int(it["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.append({"text": text, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def build_layout(
    *,
    source_width: int,
    source_height: int,
    background_image_path: str,
    ocr: Any,
    slide_height_in: float = DEFAULT_SLIDE_HEIGHT_IN,
    background_color: str = "#FFFFFF",
) -> dict:
    """Build a per-page layout dict (no I/O).

    `background_image_path` should be the path the PPT builder will
    resolve relative to its `assets_root`. The build_deck orchestrator
    points assets_root at the work-dir, so callers should pass a path
    relative to the work-dir (e.g. "inventory/page_01.clean.png").
    """
    slide_width_in = slide_height_in * (source_width / max(1, source_height))
    elements: list[dict] = [
        {
            "type": "image",
            "name": "bg",
            "path": background_image_path,
            "box": [0, 0, int(source_width), int(source_height)],
            "role": "background",
        }
    ]
    for idx, item in enumerate(_ocr_items(ocr)):
        w = item["x2"] - item["x1"]
        h = item["y2"] - item["y1"]
        pt = _font_pt(h, source_height, slide_height_in, item["text"])
        elements.append({
            "type": "text",
            "name": f"t{idx:03d}",
            "text": item["text"],
            "box": [item["x1"], item["y1"], w, h],
            "source_bbox": [item["x1"], item["y1"], item["x2"], item["y2"]],
            "font": "Microsoft YaHei",
            "size": pt,
            "bold": False,
            "color": "#111111",
            "align": "left",
            "valign": "middle",
            "line_spacing": 1.0,
        })

    return {
        "slide_size": {
            "width_in": slide_width_in,
            "height_in": slide_height_in,
        },
        "source_width": int(source_width),
        "source_height": int(source_height),
        "background": background_color,
        "elements": elements,
    }


def write_layout(
    *,
    page_num: str,
    source_width: int,
    source_height: int,
    clean_rel_path: str,
    ocr_path: Path,
    out_layout_path: Path,
    slide_height_in: float = DEFAULT_SLIDE_HEIGHT_IN,
) -> dict:
    """Read OCR, build layout, write to disk. Returns the layout dict."""
    ocr = json.loads(Path(ocr_path).read_text(encoding="utf-8"))
    layout = build_layout(
        source_width=source_width,
        source_height=source_height,
        background_image_path=clean_rel_path,
        ocr=ocr,
        slide_height_in=slide_height_in,
    )
    out_layout_path.parent.mkdir(parents=True, exist_ok=True)
    out_layout_path.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    n_text = sum(1 for e in layout["elements"] if e["type"] == "text")
    return {"page": page_num, "text": n_text, "image": 1, "tables": 0}
