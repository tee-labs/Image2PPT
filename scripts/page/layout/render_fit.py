"""Best-fit font/size/position by comparing local renders to source ink.

Renders every candidate (font, bold, size) locally via Pillow, scores
the result against the source ink mask (height + width + area + shape
error plus a bias toward the initial size and font), and returns the
best match. When the best score is still far from the source, returns
None so the caller falls back to the conservative bbox-height estimate.
"""
from __future__ import annotations

import numpy as np

from layout.render_font import _font_candidates, _render_text_metrics
from layout.render_ink import _mask_shape_error, _target_text_metrics
from layout.text_sizing import _FONT_LADDER, _is_punct_run
from layout.text_units import _estimated_text_width_px


def _binary_size_candidates(
    text: str,
    font_path: str,
    bold: bool,
    target_h: int,
    initial_size: int,
    pt_per_px: float,
) -> list[int]:
    """Fast-lock likely pt sizes by binary-searching rendered ink height."""
    lo, hi = 5, 96
    for _ in range(8):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        metrics = _render_text_metrics(text, font_path, mid, pt_per_px, bold)
        mid_h = int(metrics["ink_h"]) if metrics else 0
        if mid_h < target_h:
            lo = mid + 1
        else:
            hi = mid
    center = lo
    candidates = set(range(center - 3, center + 4))
    candidates.update(range(int(initial_size) - 4, int(initial_size) + 5))
    candidates.update(v for v in _FONT_LADDER
                      if abs(v - int(initial_size)) <= 8)
    return sorted(v for v in candidates if 5 <= v <= 96)


def fit_text_render(
    text: str,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    initial_size: int,
    initial_bold: bool,
    initial_font: str,
    pt_per_px: float,
) -> dict | None:
    """Choose font/size/position by comparing render masks to source ink."""
    if not text or "\n" in text:
        return None
    target = _target_text_metrics(source, bbox)
    if target is None or target["ink_h"] < 3 or target["ink_w"] < 3:
        return None
    fonts = _font_candidates()
    if not fonts:
        return None
    if _is_punct_run(text):
        fonts = [f for f in fonts
                 if f["ppt_name"] == initial_font] or fonts[:1]

    best: dict | None = None
    bold_options = [bool(initial_bold)]
    if text.strip() and len(text.strip()) <= 40:
        bold_options.append(not bool(initial_bold))

    for font_idx, font in enumerate(fonts):
        font_path = font["path"]
        font_penalty = float(font.get("penalty", 0.0))
        if font["ppt_name"] != initial_font:
            font_penalty += 0.02 + font_idx * 0.005
        for bold in bold_options:
            size_candidates = _binary_size_candidates(
                text, font_path, bold, int(target["ink_h"]),
                int(initial_size), pt_per_px,
            )
            for size in size_candidates:
                metrics = _render_text_metrics(text, font_path, size,
                                               pt_per_px, bold)
                if not metrics:
                    continue
                h_err = abs(metrics["ink_h"] - target["ink_h"]) / max(
                    1, target["ink_h"])
                w_err = abs(metrics["ink_w"] - target["ink_w"]) / max(
                    1, target["ink_w"])
                area_err = abs(
                    metrics["ink_area"] - target["ink_area"]) / max(
                    1, target["ink_area"])
                shape_err = _mask_shape_error(target["mask"], metrics["mask"])
                size_penalty = abs(size - int(initial_size)) / max(
                    16.0, float(initial_size))
                bold_penalty = 0.0 if bold == bool(initial_bold) else 0.06
                score = (
                    0.40 * h_err
                    + 0.30 * w_err
                    + 0.12 * min(1.5, area_err)
                    + 0.12 * shape_err
                    + 0.035 * min(2.0, size_penalty)
                    + font_penalty
                    + bold_penalty
                )
                candidate = {
                    "score": float(score),
                    "height_err": float(h_err),
                    "width_err": float(w_err),
                    "shape_err": float(shape_err),
                    "font": font["ppt_name"],
                    "font_path": font_path,
                    "size": int(size),
                    "bold": bool(bold),
                    "metrics": metrics,
                    "target": target,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None:
        return None
    # If the best render is still far from the source ink, keep the
    # conservative bbox-height fallback rather than trusting a bad match.
    if (best["score"] > 0.55
            and (best["height_err"] > 0.35 or best["width_err"] > 0.55)):
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    metrics = best["metrics"]
    target = best["target"]
    rx1, ry1, rx2, ry2 = (int(v) for v in metrics["ink_bbox"])
    tx1, ty1, tx2, ty2 = (int(v) for v in target["ink_bbox"])

    raw_x = tx1 - rx1
    raw_y = ty1 - ry1
    max_dx = max(5, int(round(0.35 * bbox_h)))
    max_dy = max(6, int(round(0.75 * bbox_h)))
    fit_x = int(round(max(x1 - max_dx, min(x1 + max_dx, raw_x))))
    fit_y = int(round(max(y1 - max_dy, min(y1 + max_dy, raw_y))))
    fit_x = max(0, min(source.shape[1] - 1, fit_x))
    fit_y = max(0, min(source.shape[0] - 1, fit_y))

    render_w = int(metrics["ink_w"])
    line_px = float(best["size"]) / max(0.01, pt_per_px)
    text_w = max(
        bbox_w + 6,
        render_w + 8,
        _estimated_text_width_px(
            text, int(best["size"]),
            pt_per_px=pt_per_px,
            bold=bool(best["bold"]),
        ) + 8,
    )
    text_w = min(int(text_w), max(1, int(source.shape[1]) - fit_x))
    text_h = max(
        bbox_h + 4,
        int(round(max(float(ry2) + 4.0, line_px * 1.35))),
    )

    return {
        "font": best["font"],
        "size": int(best["size"]),
        "bold": bool(best["bold"]),
        "box": [fit_x, fit_y, int(text_w), int(text_h)],
        "score": round(float(best["score"]), 4),
        "target_ink": [int(tx1), int(ty1), int(tx2), int(ty2)],
        "render_ink": [int(rx1), int(ry1), int(rx2), int(ry2)],
    }
