"""Icon-asset alpha + text scrub helpers.

Three responsibilities live here:

* `_ICON_PAD_ROLES` / `_SPARSE_VISUAL_ROLES` — the role tags that mark
  an inventory record as "icon-like" (used by bbox_shape padding +
  builder routing).
* `_scrub_text_boxes_from_icon_crop` — a second local fill pass that
  removes faint text residue from saturated icon backgrounds.
* `_line_art_alpha` — alpha mask for sparse stroke icons that keeps
  filled foreground islands + their enclosed holes but lets surrounding
  panel/background pixels go transparent.
* `_simple_background_strip` — used by `_pad_icon_bbox` to decide
  whether a candidate icon padding strip is plain enough to absorb.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]    # scripts/
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from icon import inpaint_region_inplace  # noqa: E402


_ICON_PAD_ROLES = {
    "small_icon",
    "preserve_visual_icon",
    "subicon",
    "badge_subicon",
    "connector",
    "line_subicon",
}
_SPARSE_VISUAL_ROLES = _ICON_PAD_ROLES | {"thin_rule", "outline"}


def _scrub_text_boxes_from_icon_crop(
    crop_bgr: np.ndarray,
    crop_box: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
    *,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Remove OCR-text residue from a clean-cropped icon asset.

    The source crop is already text-erased, but high-contrast text on
    saturated icon fills can leave a faint local texture after the
    global eraser. A second, crop-local fill keeps the icon asset
    neutral under the editable PPT text that will render above it.
    """
    if crop_bgr.size == 0 or not text_boxes:
        return crop_bgr
    x1, y1, x2, y2 = crop_box
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr
    support = None
    if alpha is not None and alpha.shape[:2] == (h, w):
        support = alpha > 16
    out = crop_bgr.copy()
    for tx1, ty1, tx2, ty2 in text_boxes:
        lx1 = max(0, tx1 - x1)
        ly1 = max(0, ty1 - y1)
        lx2 = min(w, tx2 - x1)
        ly2 = min(h, ty2 - y1)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[ly1:ly2, lx1:lx2] = True
        if support is not None:
            mask &= support
        if int(mask.sum()) < 4:
            continue

        pad = max(4, min(10, int(round(max(lx2 - lx1, ly2 - ly1) * 0.18))))
        rx1, ry1 = max(0, lx1 - pad), max(0, ly1 - pad)
        rx2, ry2 = min(w, lx2 + pad), min(h, ly2 + pad)
        ring = np.zeros((h, w), dtype=bool)
        ring[ry1:ry2, rx1:rx2] = True
        ring &= ~mask
        if support is not None:
            ring &= support
        pixels = out[ring]
        fill = None
        if len(pixels) >= 12:
            flat = pixels.reshape(-1, 3).astype(np.uint8)
            quant = (flat.astype(np.uint16) // 16).astype(np.uint8)
            keys, counts = np.unique(quant, axis=0, return_counts=True)
            key = keys[int(np.argmax(counts))]
            cluster = flat[(quant == key).all(axis=1)]
            if len(cluster) < max(8, int(0.20 * len(flat))):
                cluster = flat
            fill = np.median(cluster.reshape(-1, 3), axis=0).astype(np.uint8)
        inpaint_region_inplace(out, mask, radius=2, fill_color=fill)
    return out


def _simple_background_strip(source: np.ndarray,
                             box: tuple[int, int, int, int]) -> bool:
    """Return True when a candidate padding strip looks like plain bg.

    The strip can be white slide bg or a uniform card/bg fill. We avoid
    requiring near-white because many icons sit on coloured panels.
    """
    x1, y1, x2, y2 = box
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, int(x1)))
    x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1)))
    y2 = max(0, min(h_img, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return False
    strip = source[y1:y2, x1:x2]
    if strip.size == 0:
        return False

    flat = strip.reshape(-1, 3).astype(np.int16)
    med = np.median(flat, axis=0)
    diff = np.abs(flat - med).max(axis=1)
    stable = (
        float(np.percentile(diff, 90)) <= 22
        or float((diff <= 28).sum()) / max(1, len(diff)) >= 0.90
    )

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    light_neutral = (
        float(((gray > 238) & (hsv[:, :, 1] < 35)).sum())
        / max(1, gray.size)
        >= 0.85
    )
    return bool(stable or light_neutral)


def _line_art_alpha(crop_bgr: np.ndarray) -> np.ndarray:
    """Alpha mask for sparse line icons without keeping rectangular bg.

    The foreground can be a pure stroke icon, a white badge on a coloured
    title bar, or a filled warning triangle with white holes. We preserve
    filled foreground islands and their enclosed holes, but leave
    ordinary surrounding panel/background pixels transparent.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8)
    ring = max(1, min(4, min(h, w) // 8))
    border = np.concatenate([
        crop_bgr[:ring, :].reshape(-1, 3),
        crop_bgr[-ring:, :].reshape(-1, 3),
        crop_bgr[:, :ring].reshape(-1, 3),
        crop_bgr[:, -ring:].reshape(-1, 3),
    ])
    bg = (np.median(border, axis=0)
          if len(border) else np.array([255, 255, 255]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    diff = np.abs(crop_bgr.astype(int) - bg.astype(int)).max(axis=2)
    sat = hsv[:, :, 1]
    keep = (
        (diff > 16)
        & ((sat > 16) | (gray < 238) | (gray > 245))
    )
    keep = cv2.morphologyEx(
        keep.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ) > 0

    alpha_mask = np.zeros((h, w), dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        keep.astype(np.uint8), 8)
    total = max(1, h * w)
    for i in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[i])
        if area < max(3, int(round(0.0006 * total))):
            continue
        component = labels == i
        density = area / float(max(1, cw * ch))
        filled_component = component
        if density >= 0.24:
            comp_u8 = component.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                filled = np.zeros_like(comp_u8)
                cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
                filled_component = filled > 0
        alpha_mask |= filled_component

    alpha = np.zeros((h, w), dtype=np.uint8)
    soft = np.clip((diff.astype(int) - 8) * 14, 0, 255).astype(np.uint8)
    alpha[keep] = soft[keep]
    alpha[alpha_mask] = np.maximum(alpha[alpha_mask], 245)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha
