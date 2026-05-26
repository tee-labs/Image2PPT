"""Mixed-size run detection: split a line into differently-sized runs.

PaddleOCR returns one line bbox per OCR record. A common PPT pattern is
a larger leading label followed by a smaller parenthetical or value
suffix in the same line. A single line-level render-fit would pick a
compromise size; the detector here measures per-character ink heights
and, when a clear size step is visible, emits per-run sizes that PPT
honours.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from layout.color import _color_close
from layout.text_sizing import (
    CAPTURE_FACTOR_SHORT,
    _is_punct_run,
    _snap_to_ladder,
    font_size_pt,
)


def _glyph_height_in_char_box(source: np.ndarray,
                              char_box: list[int],
                              line_bg: np.ndarray) -> int | None:
    """Measure actual glyph ink height inside one OCR char/word box."""
    h_img, w_img = source.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in char_box)
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = source[y1:y2, x1:x2]
    if sub.size == 0:
        return None

    local_bg = line_bg
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] > 50
    if int(sat.sum()) > 0.45 * sat.size:
        local_bg = np.median(sub[sat], axis=0)
    diff = np.abs(sub.astype(int) - local_bg.astype(int)).max(axis=2)
    mask = diff > 35

    # White text on a saturated badge/header: the local bg is the fill,
    # text is the low-saturation, higher-value foreground.
    if int(sat.sum()) > 0.45 * sat.size:
        bg_hsv = cv2.cvtColor(
            np.array([[local_bg]], dtype=np.uint8),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        light = (
            (hsv[:, :, 1] < 130)
            & (hsv[:, :, 2].astype(int) > int(bg_hsv[2]) + 25)
        )
        if int(light.sum()) >= 2:
            mask = light

    if int(mask.sum()) < 2:
        return None
    row_has_ink = mask.sum(axis=1) >= 1
    rows = np.where(row_has_ink)[0]
    if rows.size < 2:
        return None
    return int(rows.max() - rows.min() + 1)


def _char_colors_from_runs(text: str, runs: list[dict] | None,
                           fallback_color: str) -> list[str]:
    colors: list[str] = []
    if runs:
        for run in runs:
            color = run.get("color") or fallback_color
            colors.extend([color] * len(str(run.get("text", ""))))
    if len(colors) != len(text):
        colors = [fallback_color] * len(text)
    # If per-character sampling produced runs but they all have the
    # same sampled colour, there is no real inline colour variation.
    elif len({c for c in colors if c}) <= 1:
        colors = [fallback_color] * len(text)
    return colors


def _build_runs_from_char_sizes(text: str, colors: list[str],
                                sizes: list[int],
                                bolds: list[bool] | None = None,
                                ) -> list[dict]:
    runs: list[dict] = []
    cur_text = ""
    cur_color: str | None = None
    cur_size: int | None = None
    cur_bold: bool | None = None
    preserve_bold = bolds is not None and len(bolds) == len(text)
    if not preserve_bold:
        bolds = [False] * len(text)
    for c, color, size, is_bold in zip(text, colors, sizes, bolds):
        if (cur_text and color == cur_color and size == cur_size
                and bool(is_bold) == bool(cur_bold)):
            cur_text += c
            continue
        if cur_text:
            run = {"text": cur_text, "color": cur_color,
                   "size": int(cur_size)}
            if preserve_bold or cur_bold:
                run["bold"] = bool(cur_bold)
            runs.append(run)
        cur_text = c
        cur_color = color
        cur_size = int(size)
        cur_bold = bool(is_bold)
    if cur_text:
        run = {"text": cur_text, "color": cur_color,
               "size": int(cur_size)}
        if preserve_bold or cur_bold:
            run["bold"] = bool(cur_bold)
        runs.append(run)
    return runs


def _label_prefix_split_index(text: str, runs: list[dict] | None,
                              fallback_color: str) -> int | None:
    """Find `coloured label:` + body split point in mixed-colour lines."""
    colon_positions = [i for i, ch in enumerate(text) if ch in "：:"]
    if not colon_positions:
        return None
    colors = _char_colors_from_runs(text, runs, fallback_color)
    if len(colors) != len(text) or len({c for c in colors if c}) <= 1:
        return None
    for colon_idx in colon_positions:
        split_idx = colon_idx + 1
        prefix = text[:split_idx]
        suffix = text[split_idx:]
        prefix_visible = [c for c in prefix if c.strip() and c not in "：:"]
        suffix_visible = [c for c in suffix if c.strip()]
        if len(prefix_visible) < 2 or len(prefix_visible) > 14:
            continue
        if len(suffix_visible) < 3:
            continue
        prefix_colors = [colors[i] for i in range(split_idx)
                         if text[i].strip() and colors[i]]
        suffix_colors = [colors[i] for i in range(split_idx, len(text))
                         if text[i].strip() and colors[i]]
        if not prefix_colors or not suffix_colors:
            continue
        prefix_color = max(set(prefix_colors), key=prefix_colors.count)
        suffix_color = max(set(suffix_colors), key=suffix_colors.count)
        if prefix_color == suffix_color or _color_close(
                prefix_color, suffix_color, tol=30):
            continue
        if prefix_colors.count(prefix_color) < max(
                2, int(0.65 * len(prefix_colors))):
            continue
        if suffix_colors.count(suffix_color) < max(
                2, int(0.55 * len(suffix_colors))):
            continue
        return split_idx
    return None


def mixed_size_runs_from_char_boxes(
    text: str,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    char_boxes: list[list[int]] | None,
    existing_runs: list[dict] | None,
    fallback_color: str,
    *,
    bold: bool,
    pt_per_px: float,
) -> dict | None:
    """Detect common same-line mixed-size patterns and emit run sizes."""
    # Lazy imports break circular dependencies: render_ink/render_fit
    # both reach back into helpers defined here.
    from layout.render_fit import fit_text_render
    from layout.render_ink import (
        _sample_background_for_bbox,
        _target_text_metrics,
    )
    from layout.render_font import default_ppt_font

    if not char_boxes or len(char_boxes) != len(text) or pt_per_px <= 0:
        return None
    open_indices = [i for i, c in enumerate(text) if c in "（("]
    split_kind = "paren"
    if open_indices:
        split_idx = open_indices[0]
    else:
        label_split = _label_prefix_split_index(
            text, existing_runs, fallback_color)
        if label_split is None:
            return None
        split_kind = "label"
        split_idx = label_split
    if split_idx < 2 or len(text) - split_idx < 3:
        return None

    line_bg = _sample_background_for_bbox(source, bbox)
    raw_heights: list[int | None] = [
        _glyph_height_in_char_box(source, cb, line_bg)
        for cb in char_boxes
    ]

    def visible_heights(indices) -> list[int]:
        vals: list[int] = []
        for i in indices:
            c = text[i]
            h = raw_heights[i]
            if h is None or not c.strip() or _is_punct_run(c):
                continue
            vals.append(int(h))
        return vals

    prefix = visible_heights(range(0, split_idx))
    suffix = visible_heights(range(split_idx, len(text)))
    if len(prefix) < 2 or len(suffix) < 2:
        return None
    prefix_h = float(np.median(prefix))
    suffix_h = float(np.median(suffix))
    if prefix_h <= 0 or suffix_h <= 0:
        return None
    # Require a visible size step. Avoids splitting lines where glyph
    # shape alone makes some characters 1-2 px shorter.
    if suffix_h > prefix_h * 0.90 or prefix_h - suffix_h < 3:
        return None

    def _union_box(boxes: list[list[int]]) -> tuple[int, int, int, int]:
        return (
            min(int(b[0]) for b in boxes),
            min(int(b[1]) for b in boxes),
            max(int(b[2]) for b in boxes),
            max(int(b[3]) for b in boxes),
        )

    prefix_text = text[:split_idx]
    suffix_text = text[split_idx:]
    prefix_box = _union_box(char_boxes[:split_idx])
    suffix_box = _union_box(char_boxes[split_idx:])

    def _initial_size(span_text: str,
                      span_box: tuple[int, int, int, int],
                      glyph_h: float) -> int:
        if split_kind == "label":
            raw = glyph_h * pt_per_px * CAPTURE_FACTOR_SHORT
            return _snap_to_ladder(max(8.0, min(36.0, raw)))
        return font_size_pt(span_text,
                            span_box[2] - span_box[0],
                            span_box[3] - span_box[1],
                            pt_per_px=pt_per_px)

    prefix_init = _initial_size(prefix_text, prefix_box, prefix_h)
    suffix_init = _initial_size(suffix_text, suffix_box, suffix_h)

    def _mostly_cjk(value: str) -> bool:
        chars = [c for c in value if c.strip() and not _is_punct_run(c)]
        if not chars:
            return False
        cjk = sum(1 for c in chars if 0x4E00 <= ord(c) <= 0x9FFF)
        return cjk >= max(2, int(math.ceil(len(chars) * 0.70)))

    def _ink_density(box: tuple[int, int, int, int],
                     glyph_h: float) -> float | None:
        if glyph_h <= 0:
            return None
        metrics = _target_text_metrics(source, box)
        if metrics is None or metrics["ink_w"] <= 0:
            return None
        return float(metrics["ink_area"]) / max(
            1.0, float(metrics["ink_w"]) * float(glyph_h))

    prefix_bold = bool(bold)
    suffix_bold = bool(bold)
    if split_kind == "label":
        prefix_bold = _mostly_cjk(prefix_text)
        suffix_bold = False
    prefix_density = _ink_density(prefix_box, prefix_h)
    if (not prefix_bold
            and _mostly_cjk(prefix_text)
            and prefix_h >= 10
            and prefix_density is not None
            and prefix_density >= 0.52):
        prefix_bold = True

    # Look up fit_text_render via the facade so tests can patch it
    # (`itl.fit_text_render = fake_fit`).
    import inventory_to_layout as _itl
    fit = _itl.fit_text_render

    prefix_fit = fit(
        prefix_text, source, prefix_box,
        initial_size=prefix_init,
        initial_bold=prefix_bold,
        initial_font=default_ppt_font(),
        pt_per_px=pt_per_px,
    )
    suffix_fit = fit(
        suffix_text, source, suffix_box,
        initial_size=suffix_init,
        initial_bold=suffix_bold,
        initial_font=default_ppt_font(),
        pt_per_px=pt_per_px,
    )

    base_size = int(prefix_fit["size"]) if prefix_fit else int(prefix_init)
    suffix_size = int(suffix_fit["size"]) if suffix_fit else int(suffix_init)
    if prefix_fit:
        prefix_bold = bool(prefix_fit["bold"])
    if suffix_fit:
        suffix_bold = bool(suffix_fit["bold"])
    if base_size - suffix_size < 2:
        return None

    sizes = [base_size if i < split_idx else suffix_size
             for i in range(len(text))]
    bolds = [prefix_bold if i < split_idx else suffix_bold
             for i in range(len(text))]
    colors = _char_colors_from_runs(text, existing_runs, fallback_color)
    return {
        "runs": _build_runs_from_char_sizes(text, colors, sizes, bolds),
        "base_size": base_size,
        "suffix_size": suffix_size,
        "prefix_h": round(prefix_h, 2),
        "suffix_h": round(suffix_h, 2),
        "prefix_bold": bool(prefix_bold),
        "suffix_bold": bool(suffix_bold),
        "prefix_density": (
            round(float(prefix_density), 4)
            if prefix_density is not None else None
        ),
        "prefix_fit_score": (
            round(float(prefix_fit["score"]), 4) if prefix_fit else None
        ),
        "suffix_fit_score": (
            round(float(suffix_fit["score"]), 4) if suffix_fit else None
        ),
    }
