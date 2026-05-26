"""Per-element bbox shape adjustments.

* `_pad_icon_bbox` — loosen tight detection boxes through plain
  background so icon assets don't clip antialiasing.
* `_trim_short_stat_text_bbox` — when OCR pulls a leading clock /
  pin / currency icon into a `7分钟`-style stat box, push the bbox
  rightward so the editable text doesn't sit on top of the icon.
"""
from __future__ import annotations

import cv2
import numpy as np

from layout.icon_alpha import _ICON_PAD_ROLES, _simple_background_strip
from shared.geometry import overlaps_text as _overlaps_text


def _pad_icon_bbox(
    bbox: tuple[int, int, int, int],
    probe_img: np.ndarray,
    text_boxes: list[tuple[int, int, int, int]],
    role: str | None,
    *,
    allow_text_overlap: bool = False,
) -> tuple[int, int, int, int]:
    """Loosen icon-ish crops when the surrounding pixels are background.

    Tight detection boxes often clip antialiasing or make the resulting
    PPT image object awkwardly precise. Expanding only through plain
    background avoids baking nearby editable text into the asset. When
    the crop source is already text-erased, OCR text boxes can be
    ignored because the text shapes render above every image later.
    """
    h_img, w_img = probe_img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(w_img, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1))
    y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x2, y2)

    w, h = x2 - x1, y2 - y1
    explicit_icon = role in _ICON_PAD_ROLES
    implicit_cv2_icon = role in {None, ""}
    if not explicit_icon and not implicit_cv2_icon:
        return (x1, y1, x2, y2)

    # Keep this scoped to icon-ish elements. Medium icons can safely
    # receive a little more background when cropped from a text-erased
    # source; very large regions are probably cards/screenshots.
    max_side = 340 if allow_text_overlap else 280
    max_area = 80000 if allow_text_overlap else 52000
    if w > max_side or h > max_side or w * h > max_area:
        return (x1, y1, x2, y2)
    ratio = 0.06 if max(w, h) > 140 else 0.08
    pad_max = 10 if allow_text_overlap else 8
    pad = max(2, min(pad_max, int(round(max(w, h) * ratio))))
    if role == "subicon":
        pad = min(pad, 6)

    nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    if x1 > 0:
        px1 = max(0, x1 - pad)
        side = (px1, y1, x1, y2)
        cand = (px1, y1, x2, y2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            nx1 = px1
    if x2 < w_img:
        px2 = min(w_img, x2 + pad)
        side = (x2, y1, px2, y2)
        cand = (nx1, y1, px2, y2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            nx2 = px2
    if y1 > 0:
        py1 = max(0, y1 - pad)
        side = (nx1, py1, nx2, y1)
        cand = (nx1, py1, nx2, y2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            ny1 = py1
    if y2 < h_img:
        py2 = min(h_img, y2 + pad)
        side = (nx1, y2, nx2, py2)
        cand = (nx1, ny1, nx2, py2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            ny2 = py2

    out = (nx1, ny1, nx2, ny2)
    if not allow_text_overlap and _overlaps_text(out, text_boxes):
        return (x1, y1, x2, y2)
    return out


def _trim_short_stat_text_bbox(
    text: str,
    bbox: tuple[int, int, int, int],
    source: np.ndarray,
) -> tuple[int, int, int, int]:
    """Shrink OCR boxes that include a leading icon before a short stat.

    OCR often reports `7分钟` / `16小时` together with the clock icon on
    the left. The eraser preserves that icon; the editable text box
    must start after it or the rendered text will sit on top of the icon.
    """
    if len(text) > 5 or not any(c.isdigit() for c in text):
        return bbox
    compact = "".join(c for c in text if not c.isspace())
    if not compact:
        return bbox
    # If the text starts with a real word glyph (`前10名`, `第4作者`),
    # the leading component is part of the text and stays in the bbox.
    if not (compact[0].isdigit() or compact[0] in "+-￥¥$€£"):
        return bbox
    x1, y1, x2, y2 = bbox
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return bbox
    region = source[y1:y2, x1:x2]
    h, w = region.shape[:2]
    # All pixel thresholds in this function are calibrated against a
    # 720-tall reference image. `scale` rescales them.
    scale = h_img / 720.0
    if h < 20 * scale or w < 40 * scale:
        return bbox
    border = np.concatenate([
        region[:3, :].reshape(-1, 3),
        region[-3:, :].reshape(-1, 3),
        region[:, :3].reshape(-1, 3),
        region[:, -3:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    diff = np.abs(region.astype(int) - bg.astype(int)).max(axis=2)
    fg = cv2.morphologyEx((diff > 30).astype(np.uint8) * 255,
                          cv2.MORPH_CLOSE,
                          np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    col_fg = fg.sum(axis=0)
    min_area = 80 * scale * scale
    min_dim = max(3, int(round(3 * scale)))
    max_icon_dim = 75 * scale
    gap_window = max(12, int(round(12 * scale)))
    min_gap_run = max(3, int(round(3 * scale)))
    trim_to = x1
    for i in range(1, n):
        lx, ly, lw, lh, area = (int(v) for v in stats[i])
        if area < min_area or lw <= min_dim or lh <= min_dim:
            continue
        aspect = lw / max(1, lh)
        touches_left_icon_zone = lx <= max(4 * scale, int(0.12 * w))
        # Real glyph-prefix icons (clock, pin, $, ¥) are close to
        # square — aspect ≥ 0.60. Leading digits like the `1` in
        # `1亿人` are tall and narrow; the 0.60 lower bound keeps them
        # out of the icon path.
        iconish = (
            touches_left_icon_zone
            and lh >= 0.35 * h
            and max(lw, lh) <= max_icon_dim
            and 0.60 <= aspect <= 1.60
        )
        if not iconish:
            continue
        # Even when the leading CC looks square-ish, require a clear
        # horizontal gap (≥ ~3 px @ 720-ref) right after it. Genuine
        # icon-then-text has that gap; a wide bold digit accidentally
        # hitting the aspect window would have its next glyph adjacent.
        gap_start = lx + lw
        gap_end = min(w, gap_start + gap_window)
        if gap_end <= gap_start:
            continue
        run = 0
        ok = False
        for c in range(gap_start, gap_end):
            if col_fg[c] == 0:
                run += 1
                if run >= min_gap_run:
                    ok = True
                    break
            else:
                run = 0
        if not ok:
            continue
        trim_to = max(trim_to, x1 + lx + lw + int(round(4 * scale)))
    if trim_to <= x1 + 4 * scale or trim_to >= x2 - 12 * scale:
        return bbox
    return (trim_to, y1, x2, y2)
