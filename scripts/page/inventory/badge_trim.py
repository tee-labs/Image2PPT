"""Trim connector tails off a filled badge / icon bbox.

Sparse line-art detection can grow from a filled circular node into its
attached dashed connector tails. This pass keeps the dense badge as one
icon and lets the connector records own the line segments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PAGE_DIR = Path(__file__).resolve().parents[1]
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from _heuristics import s_area, s_kernel, s_length  # noqa: E402


def trim_filled_badge(
    icon_probe: np.ndarray,
    ocr_text_items: list[dict],
    scale: float,
    x1: int, y1: int, x2: int, y2: int,
    parent_mask: np.ndarray,
    parent_origin: tuple[int, int],
) -> tuple[int, int, int, int, np.ndarray]:
    """Trim a filled badge's bbox down to just the badge carrier."""
    crop = icon_probe[y1:y2, x1:x2]
    if crop.size == 0:
        return x1, y1, x2, y2, parent_mask
    h, w = crop.shape[:2]
    if max(w, h) < s_length(55, scale):
        return x1, y1, x2, y2, parent_mask
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    dense = (gray < 170) | ((hsv[:, :, 1] > 75) & (gray < 225))
    kernel_size = max(5, s_kernel(9, scale))
    if kernel_size % 2 == 0:
        kernel_size += 1
    dense_u8 = cv2.morphologyEx(
        dense.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dense_u8, 8)
    best = None
    best_label = -1
    best_area = 0
    min_area = s_area(900, scale)
    min_side = s_length(34, scale)
    for i in range(1, n):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area or bw < min_side or bh < min_side:
            continue
        aspect = bw / float(max(1, bh))
        density = area / float(max(1, bw * bh))
        if not (0.55 <= aspect <= 1.65 and density >= 0.38):
            continue
        if area > best_area:
            best = (bx, by, bx + bw, by + bh)
            best_label = i
            best_area = area
    if best is None:
        return x1, y1, x2, y2, parent_mask

    bx1, by1, bx2, by2 = best
    selected = labels == best_label
    recover_size = max(3, s_kernel(5, scale))
    if recover_size % 2 == 0:
        recover_size += 1
    selected = cv2.dilate(
        selected.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (recover_size, recover_size)),
        iterations=1,
    ).astype(bool)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    carrier_aspect = bw / float(bh)
    if 0.72 <= carrier_aspect <= 1.38:
        carrier_clip = np.zeros((h, w), dtype=np.uint8)
        center = (int(round((bx1 + bx2) / 2.0)),
                  int(round((by1 + by2) / 2.0)))
        axes = (max(1, int(round(bw / 2.0))),
                max(1, int(round(bh / 2.0))))
        cv2.ellipse(carrier_clip, center, axes, 0, 0, 360, 255, -1)
        selected &= carrier_clip.astype(bool)

    raw_u8 = dense.astype(np.uint8) * 255
    rn, rlabels, rstats, _ = cv2.connectedComponentsWithStats(raw_u8, 8)
    # Include small badge adornments directly below the main carrier
    # (e.g. a lock hanging from a telecom node), but not long diagonal
    # connector tails.
    ux1, uy1, ux2, uy2 = bx1, by1, bx2, by2
    for i in range(1, rn):
        cx, cy, cw, ch, area = (int(v) for v in rstats[i])
        if area < s_area(40, scale):
            continue
        ccx = cx + cw / 2.0
        ccy = cy + ch / 2.0
        close_below = (
            bx1 - s_length(18, scale) <= ccx <= bx2 + s_length(18, scale)
            and by1 - s_length(8, scale) <= ccy <= by2 + s_length(36, scale)
        )
        if close_below:
            ux1, uy1 = min(ux1, cx), min(uy1, cy)
            ux2, uy2 = max(ux2, cx + cw), max(uy2, cy + ch)
            selected |= rlabels == i

    text_mask = np.zeros((h, w), dtype=bool)
    for it in ocr_text_items:
        tx1 = max(0, int(it["x1"]) - x1)
        ty1 = max(0, int(it["y1"]) - y1)
        tx2 = min(w, int(it["x2"]) - x1)
        ty2 = min(h, int(it["y2"]) - y1)
        if tx2 > tx1 and ty2 > ty1:
            text_mask[ty1:ty2, tx1:tx2] = True
    near_selected = cv2.dilate(
        selected.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (recover_size, recover_size)),
        iterations=1,
    ).astype(bool)
    light_glyph = (gray > 210) & (hsv[:, :, 1] < 80) & ~text_mask
    selected |= light_glyph & near_selected

    px0, py0 = parent_origin
    lx1 = max(0, x1 - px0)
    ly1 = max(0, y1 - py0)
    lx2 = min(parent_mask.shape[1], x2 - px0)
    ly2 = min(parent_mask.shape[0], y2 - py0)
    selected &= parent_mask[ly1:ly2, lx1:lx2].astype(bool)

    pad = max(1, s_length(2, scale))
    nx1 = max(0, ux1 - pad)
    ny1 = max(0, uy1 - pad)
    nx2 = min(w, ux2 + pad)
    ny2 = min(h, uy2 + pad)
    old_area = max(1, w * h)
    new_area = max(1, (nx2 - nx1) * (ny2 - ny1))
    child_mask = np.zeros_like(parent_mask, dtype=bool)
    child_mask[ly1:ly2, lx1:lx2] = selected
    if int(child_mask.sum()) < s_area(40, scale):
        return x1, y1, x2, y2, parent_mask
    if new_area >= 0.92 * old_area:
        return x1, y1, x2, y2, child_mask
    return x1 + nx1, y1 + ny1, x1 + nx2, y1 + ny2, child_mask
