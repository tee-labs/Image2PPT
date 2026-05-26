"""Background sampling + target-ink mask measurement.

`sample_background_for_bbox` is re-exported from `shared.bg_sample` so
mixed_size_runs etc. can reach it through a stable in-package name.
`_target_text_metrics` produces the binary ink mask that the render-fit
search compares against locally-rendered candidates.
"""
from __future__ import annotations

import cv2
import numpy as np

from shared.bg_sample import sample_background_for_bbox as _sample_background_for_bbox  # noqa: F401


def _target_text_metrics(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict | None:
    """Measure the actual source ink inside/near an OCR text bbox."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    bg = _sample_background_for_bbox(source, (x1, y1, x2, y2))
    pad = max(1, min(3, int(round((y2 - y1) * 0.12))))
    px1 = max(0, x1 - pad); px2 = min(w_img, x2 + pad)
    py1 = max(0, y1 - pad); py2 = min(h_img, y2 + pad)
    region = source[py1:py2, px1:px2]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > 50
    total_px = region.shape[0] * region.shape[1]
    is_container_like = int(sat_mask.sum()) > 0.55 * total_px
    container_area = None
    largest_component = None
    if is_container_like:
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
            sat_mask.astype(np.uint8) * 255, 8)
        if n_comp > 1:
            largest_rel = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            largest_label = largest_rel + 1
            largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
            total_sat = int(sat_mask.sum())
            largest_component = (labels == largest_label).astype(np.uint8)
            if largest_area < 0.55 * total_sat:
                is_container_like = False
        else:
            largest_component = sat_mask.astype(np.uint8)
    if is_container_like:
        container_bg = np.median(region[sat_mask], axis=0)
        if int(np.max(np.abs(
                container_bg.astype(int) - bg.astype(int)))) > 40:
            bg = container_bg
            component = (largest_component
                         if largest_component is not None
                         else sat_mask.astype(np.uint8))
            contours, _ = cv2.findContours(
                component * 255, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros(component.shape, dtype=np.uint8)
            if contours:
                cv2.drawContours(filled, contours, -1, 255, thickness=-1)
                container_area = filled.astype(bool)
            else:
                container_area = component.astype(bool)
    diff = np.abs(region.astype(int) - bg.astype(int)).max(axis=2)
    mask = diff > 35
    if container_area is not None:
        bg_hsv = cv2.cvtColor(
            np.array([[bg]], dtype=np.uint8),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        light_text = (
            container_area
            & (hsv[:, :, 1] < 120)
            & (hsv[:, :, 2].astype(int) > int(bg_hsv[2]) + 28)
        )
        if int(light_text.sum()) >= 5:
            mask = light_text
            full_rows = mask.sum(axis=1) > 0.75 * mask.shape[1]
            full_cols = mask.sum(axis=0) > 0.75 * mask.shape[0]
            mask[full_rows, :] = False
            mask[:, full_cols] = False
        else:
            mask &= container_area
            full_rows = mask.sum(axis=1) > 0.75 * mask.shape[1]
            full_cols = mask.sum(axis=0) > 0.75 * mask.shape[0]
            mask[full_rows, :] = False
            mask[:, full_cols] = False
    if int(mask.sum()) < 5:
        thr = max(15, int(np.percentile(diff, 75)))
        mask = diff > thr
    if int(mask.sum()) < 5:
        return None

    # Reject sparse antialias dust while keeping thin 8-9 pt CJK text.
    min_row_ink = 1 if (px2 - px1) <= 18 else 2
    row_has_ink = mask.sum(axis=1) >= min_row_ink
    col_has_ink = mask.sum(axis=0) >= 1
    rows = np.where(row_has_ink)[0]
    cols = np.where(col_has_ink)[0]
    if rows.size < 2 or cols.size < 2:
        return None

    ix1 = int(cols.min()); ix2 = int(cols.max()) + 1
    iy1 = int(rows.min()); iy2 = int(rows.max()) + 1
    mask_crop = mask[iy1:iy2, ix1:ix2]
    return {
        "ink_bbox": [px1 + ix1, py1 + iy1, px1 + ix2, py1 + iy2],
        "ink_w": int(ix2 - ix1),
        "ink_h": int(iy2 - iy1),
        "ink_area": int(mask_crop.sum()),
        "mask": mask_crop,
    }


def _mask_shape_error(target_mask: np.ndarray,
                      render_mask: np.ndarray) -> float:
    if target_mask.size == 0 or render_mask.size == 0:
        return 0.5
    th, tw = target_mask.shape[:2]
    if th < 3 or tw < 3:
        return 0.0
    resized = cv2.resize(
        render_mask.astype(np.uint8) * 255, (tw, th),
        interpolation=cv2.INTER_AREA) > 64
    target = target_mask.astype(bool)
    inter = int((target & resized).sum())
    union = int((target | resized).sum())
    if union <= 0:
        return 0.5
    return 1.0 - (inter / union)
