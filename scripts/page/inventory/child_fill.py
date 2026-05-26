"""Child-fill quality assessment used by every shape-extraction path.

When the inventory builder wants to remove a nested icon / shape from
its parent and emit it as a separate movable element, it must be sure
the patched parent will blend cleanly with the surrounding pixels. The
three helpers here are deliberately conservative: if the ring around
the removal mask is mixed (gradient/photo/nearby artwork) we keep the
child baked into the parent instead of leaving a visible scar.
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

PAGE_DIR = Path(__file__).resolve().parents[1]
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from _heuristics import s_area  # noqa: E402


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ring_bg_quality(crop: np.ndarray, mask: np.ndarray, scale: float,
                    *,
                    fallback_color: np.ndarray | None = None,
                    ) -> tuple[bool, np.ndarray]:
    """Check whether a child can be erased cleanly from its parent."""
    if crop.size == 0 or mask.size == 0 or int(mask.sum()) < 4:
        color = (fallback_color if fallback_color is not None
                 else np.array([255, 255, 255], dtype=np.uint8))
        return False, color.astype(np.uint8)
    h, w = mask.shape[:2]
    bbox = _mask_bbox(mask)
    if bbox is None:
        color = (fallback_color if fallback_color is not None
                 else np.array([255, 255, 255], dtype=np.uint8))
        return False, color.astype(np.uint8)
    x1, y1, x2, y2 = bbox
    pad = max(3, int(round(7 * scale)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
    if rx2 <= rx1 or ry2 <= ry1:
        color = (fallback_color if fallback_color is not None
                 else np.array([255, 255, 255], dtype=np.uint8))
        return False, color.astype(np.uint8)
    local = crop[ry1:ry2, rx1:rx2]
    local_mask = mask[ry1:ry2, rx1:rx2]
    ring = np.ones(local_mask.shape, dtype=bool)
    ring[local_mask] = False
    pixels = local[ring]
    if len(pixels) < max(12, s_area(20, scale)):
        color = (fallback_color if fallback_color is not None
                 else np.array([255, 255, 255], dtype=np.uint8))
        return False, color.astype(np.uint8)

    flat = pixels.reshape(-1, 3).astype(np.uint8)
    quant = (flat.astype(np.uint16) // 16).astype(np.uint8)
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    key = keys[int(np.argmax(counts))]
    cluster = flat[(quant == key).all(axis=1)]
    if len(cluster) < max(8, int(0.18 * len(flat))):
        cluster = flat
    bg = np.median(cluster.reshape(-1, 3), axis=0).astype(np.uint8)
    diff = np.abs(flat.astype(np.int16) - bg.astype(np.int16)).max(axis=1)
    uniform = (
        float(np.percentile(diff, 90)) <= 38
        or int((diff <= 45).sum()) >= 0.82 * len(diff)
    )
    if fallback_color is not None:
        close_to_fill = (
            int(np.max(np.abs(
                bg.astype(np.int16) - fallback_color.astype(np.int16)))) <= 55
        )
        uniform = uniform and close_to_fill
    return bool(uniform), bg


def simulated_fill_quality(crop: np.ndarray, mask: np.ndarray,
                           fill_color: np.ndarray, scale: float) -> bool:
    """Verify the patched parent would blend into its surround."""
    if crop.size == 0 or mask.size == 0 or int(mask.sum()) < 4:
        return False
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    patch_u8 = mask.astype(np.uint8) * 255
    patch_u8 = cv2.dilate(
        patch_u8, np.ones((3, 3), np.uint8),
        iterations=max(1, int(round(5 * scale))),
    )
    patch_mask = patch_u8 > 0
    if int(patch_mask.sum()) < 4:
        return False
    ring_u8 = cv2.dilate(
        patch_u8, np.ones((3, 3), np.uint8),
        iterations=max(2, int(round(4 * scale))),
    )
    ring = (ring_u8 > 0) & ~patch_mask
    if int(ring.sum()) < max(12, s_area(20, scale)):
        ring = ~patch_mask
    if int(ring.sum()) == 0:
        return False

    patched = crop.copy()
    inpaint_region_inplace(
        patched, mask, scale=scale, fill_color=fill_color)
    ring_pixels = patched[ring].reshape(-1, 3)
    patch_pixels = patched[patch_mask].reshape(-1, 3)
    if len(ring_pixels) == 0 or len(patch_pixels) == 0:
        return False
    bg = np.median(ring_pixels, axis=0).astype(np.int16)
    patch_diff = np.max(
        np.abs(patch_pixels.astype(np.int16) - bg[None, :]), axis=1)
    ring_diff = np.max(
        np.abs(ring_pixels.astype(np.int16) - bg[None, :]), axis=1)
    patch_p90 = float(np.percentile(patch_diff, 90))
    ring_p90 = float(np.percentile(ring_diff, 90))
    if patch_p90 > max(42.0, ring_p90 + 20.0):
        return False
    return (int((patch_diff <= max(48.0, ring_p90 + 24.0)).sum())
            >= 0.86 * len(patch_diff))


def clean_child_fill(crop: np.ndarray, mask: np.ndarray, scale: float,
                     fill_color: np.ndarray | None = None,
                     ) -> tuple[bool, np.ndarray]:
    """Return a safe fill colour for removing a child from its parent.

    The detector's suggested fill can be wrong for composite parents
    (e.g. a blue title bar embedded in a mostly white card). Prefer the
    local ring colour, then simulate the inpaint and reject the child if
    the patched pixels do not blend with the parent around the hole.
    """
    ok, bg = ring_bg_quality(crop, mask, scale)
    if not ok and fill_color is not None:
        ok, bg = ring_bg_quality(
            crop, mask, scale, fallback_color=fill_color)
    if not ok:
        return False, bg
    fill = bg.astype(np.uint8)
    if fill_color is not None:
        raw = np.asarray(fill_color, dtype=np.uint8).reshape(-1)[:3]
        if int(np.max(np.abs(
                raw.astype(np.int16) - fill.astype(np.int16)))) <= 36:
            fill = raw.astype(np.uint8)
    if not simulated_fill_quality(crop, mask, fill, scale):
        return False, fill
    return True, fill
