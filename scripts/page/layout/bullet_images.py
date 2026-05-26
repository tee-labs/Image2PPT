"""Restore stripped bullet markers as tiny transparent image assets.

After `strip_leading_list_markers` removes bullet glyphs from editable
text, the visual bullets must still appear in the rendered deck. This
module detects each bullet glyph in the source image, extracts it as an
alpha-keyed PNG, and emits an image element so the bullet renders in
front of the slide.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from layout.render_ink import _sample_background_for_bbox


def _detect_leading_dot_marker(
    el: dict,
    source: np.ndarray,
) -> dict | None:
    """Find a small dot-like component just before a text line.

    Colour-agnostic on purpose: bullets can be grey, blue, white, or any
    theme colour. False positives are controlled later by requiring
    repeated marker/body indentation across neighbouring rows.
    """
    bbox = el.get("source_bbox")
    line_h = 12.0
    body_x = None
    line_y = None
    if bbox and len(bbox) == 4:
        bx1, by1, _bx2, by2 = (float(v) for v in bbox)
        line_h = max(4.0, by2 - by1)
        body_x = bx1
        line_y = (by1 + by2) / 2.0
    boxes = el.get("source_char_boxes") or []
    if boxes:
        body_x = float(min(int(b[0]) for b in boxes if len(b) == 4))
    if el.get("ignored_marker_box"):
        marker_box = [int(v) for v in el["ignored_marker_box"]]
        marker_x = (marker_box[0] + marker_box[2]) / 2.0
        return {
            "box": marker_box,
            "forced": True,
            "marker_x": marker_x,
            "body_x": float(
                body_x if body_x is not None
                else marker_box[2] + line_h),
            "line_y": float(
                line_y if line_y is not None
                else (marker_box[1] + marker_box[3]) / 2.0),
            "line_h": float(line_h),
        }
    if not bbox or len(bbox) != 4:
        return None
    text = str(el.get("text") or "").strip()
    if len(text) < 2:
        return None
    size = float(el.get("size") or 0)
    if size > 24:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    line_h = max(4, y2 - y1)
    body_x = int(body_x if body_x is not None else x1)

    sx1 = max(0, min(x1, body_x - int(round(line_h * 1.7))))
    sx2 = max(sx1 + 1, min(w_img, body_x - 1))
    sy1 = max(0, y1 - int(round(line_h * 0.35)))
    sy2 = min(h_img, y2 + int(round(line_h * 0.35)))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    region = source[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return None
    bg = _sample_background_for_bbox(source, (sx1, sy1, sx2, sy2))
    diff = np.abs(region.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = diff > 24
    if int(mask.sum()) < 2:
        return None

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if count <= 1:
        return None

    line_cy = (y1 + y2) / 2.0
    candidates: list[tuple[float, list[int]]] = []
    for label in range(1, count):
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 1:
            continue
        if cw > max(6, line_h * 0.55) or ch > max(7, line_h * 0.70):
            continue
        if cw < 1 or ch < 1:
            continue
        aspect = cw / max(1.0, float(ch))
        if aspect < 0.35 or aspect > 2.4:
            continue
        fill_ratio = area / max(1.0, float(cw * ch))
        if fill_ratio < 0.12:
            continue
        gx1 = sx1 + cx
        gy1 = sy1 + cy
        gx2 = sx1 + cx + cw
        gy2 = sy1 + cy + ch
        comp_cy = (gy1 + gy2) / 2.0
        if abs(comp_cy - line_cy) > max(5.0, line_h * 0.45):
            continue
        # Prefer the dot nearest to the body start, but still left of it.
        if gx2 > body_x - max(3.0, line_h * 0.15):
            continue
        score = abs(comp_cy - line_cy) + 0.20 * abs(body_x - gx2)
        candidates.append((score, [gx1, gy1, gx2, gy2]))
    if not candidates:
        return None
    marker_box = min(candidates, key=lambda x: x[0])[1]
    return {
        "box": marker_box,
        "forced": False,
        "marker_x": (marker_box[0] + marker_box[2]) / 2.0,
        "body_x": float(body_x),
        "line_y": float(line_cy),
        "line_h": float(line_h),
    }


def _marker_has_neighbour(idx: int,
                          candidates: list[tuple[dict, dict]]) -> bool:
    el, info = candidates[idx]
    for j, (_other_el, other) in enumerate(candidates):
        if j == idx:
            continue
        line_h = max(float(info.get("line_h") or 1.0),
                     float(other.get("line_h") or 1.0))
        body_close = abs(
            float(info["body_x"]) - float(other["body_x"])) <= max(
                12.0, line_h * 0.90)
        marker_close = abs(
            float(info["marker_x"]) - float(other["marker_x"])) <= max(
                10.0, line_h * 0.75)
        y_gap = abs(float(info["line_y"]) - float(other["line_y"]))
        if body_close and marker_close and y_gap <= max(48.0, line_h * 3.3):
            return True
    return False


def _marker_rgba_from_source(
        source: np.ndarray,
        bbox: list[int]) -> tuple[np.ndarray, list[int]] | None:
    h_img, w_img = source.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pad = 3
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    bg = _sample_background_for_bbox(source, (x1, y1, x2, y2))
    diff = np.abs(crop.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    alpha = np.clip((diff.astype(np.int16) - 8) * 14,
                    0, 255).astype(np.uint8)
    if int((alpha > 12).sum()) < 1:
        return None
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rgba = np.dstack([crop, alpha])
    return rgba, [x1, y1, x2 - x1, y2 - y1]


def restore_ignored_bullet_marker_images(
    text_records: list[dict],
    source: np.ndarray,
    asset_dir: Path,
    asset_prefix: str,
) -> list[dict]:
    """Emit stripped/left-gutter bullet markers as tiny transparent images."""
    out: list[dict] = []
    candidates: list[tuple[dict, dict]] = []
    for el in text_records:
        info = _detect_leading_dot_marker(el, source)
        if info is None:
            continue
        candidates.append((el, info))

    keep: list[tuple[dict, dict]] = []
    for idx, (el, info) in enumerate(candidates):
        if (bool(info.get("forced"))
                or _marker_has_neighbour(idx, candidates)):
            keep.append((el, info))

    for el, info in keep:
        marker_box = [int(v) for v in info["box"]]
        rgba_result = _marker_rgba_from_source(source, marker_box)
        if rgba_result is None:
            continue
        rgba, box = rgba_result
        name = f"{el.get('name', 'text')}_bullet_marker.png"
        path = asset_dir / name
        cv2.imwrite(str(path), rgba)
        out.append({
            "type": "image",
            "name": f"{el.get('name', 'text')}_bullet_marker",
            "path": f"{asset_prefix}/{name}",
            "box": [int(v) for v in box],
            "role": "bullet_marker",
        })
        el["ignored_marker_image"] = {
            "box": [int(v) for v in box],
            "path": f"{asset_prefix}/{name}",
        }
    return out
