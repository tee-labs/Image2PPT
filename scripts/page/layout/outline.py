"""Outline / card shape helpers.

When a detected component looks like a card border (outline mask
exists), these helpers decide whether to:
  * emit it as a native PPT shape (rounded rectangle, oval, ...), or
  * keep the original alpha-masked image crop, or
  * split a tall outline into multiple stacked sub-cards.

The same module also hosts the solid-primitive classifier that lifts
small filled circles / rects (badges, colour chips) out of the PNG
crop path into editable native shapes.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from shared.geometry import bgr_to_hex as _bgr_to_hex


def _sample_outline_color(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask_path: str | None,
) -> str:
    """Sample a card/frame line colour from its alpha mask when present."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return "#C8D7EA"
    crop = source[y1:y2, x1:x2]
    pixels = None
    if mask_path and Path(mask_path).exists():
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape[:2] == source.shape[:2]:
            opaque = m[y1:y2, x1:x2] > 16
            if int(opaque.sum()) >= 8:
                pixels = crop[opaque]
    if pixels is None or len(pixels) == 0:
        band = max(2, min(6, min(y2 - y1, x2 - x1) // 18))
        edge = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        edge[:band, :] = True
        edge[-band:, :] = True
        edge[:, :band] = True
        edge[:, -band:] = True
        pixels = crop[edge]
    if len(pixels) == 0:
        return "#C8D7EA"
    med = np.median(pixels.reshape(-1, 3), axis=0)
    # If the mask sampled mostly white interior pixels, fall back to
    # the deck's common pale blue frame colour instead of emitting
    # invisible white lines.
    if int(np.max(np.abs(med.astype(int) - 255))) <= 8:
        return "#C8D7EA"
    return _bgr_to_hex(med.astype(int))


def _sample_card_fill_color(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> str:
    """Sample a pale card/panel fill colour from inside an outline bbox."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return "#FFFFFF"
    w = x2 - x1
    h = y2 - y1
    pad = max(6, min(18, min(w, h) // 10))
    inner = source[y1 + pad:y2 - pad, x1 + pad:x2 - pad]
    if inner.size == 0:
        inner = source[y1:y2, x1:x2]
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    light_bg = (gray > 220) & (hsv[:, :, 1] < 70)
    pixels = inner[light_bg]
    if len(pixels) < max(20, int(0.05 * inner.shape[0] * inner.shape[1])):
        pixels = inner.reshape(-1, 3)
    med = np.median(pixels.reshape(-1, 3), axis=0).astype(int)
    return _bgr_to_hex(med)


def _outline_should_be_native_shape(
        bbox: tuple[int, int, int, int]) -> bool:
    """Use native round-rect directly for wide/tall card-like outlines.

    Near-square outline masks need ``classify_outline_ring`` first: they
    are often circles or rings, and an unverified rounded rectangle
    drawn over a circular ring creates an extra visible box.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / float(h)
    return aspect >= 1.45 or aspect <= 0.69


def _outline_should_keep_full_crop(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Keep a filled card as an image crop when native shapes are disabled."""
    fill = _sample_card_fill_color(source, bbox).lstrip("#")
    if len(fill) != 6:
        return False
    rgb = np.array([int(fill[i:i + 2], 16) for i in (0, 2, 4)], dtype=int)
    return int(np.max(np.abs(rgb - 255))) > 7


def _split_filled_outline_rows(
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Split stacked filled sub-cards inside one detected outline bbox."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    crop = source[y1:y2, x1:x2]
    if crop.size == 0:
        return [(x1, y1, x2, y2)]
    h, w = crop.shape[:2]
    if w < 220 or h < 90:
        return [(x1, y1, x2, y2)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(int) - 255).max(axis=2)
    panel = (
        ((gray < 253) & (diff_white > 4))
        | ((hsv[:, :, 1] > 6) & (diff_white > 3))
    )
    row_count = panel.sum(axis=1)
    active = row_count > max(18, int(0.08 * w))

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    gap_start: int | None = None
    min_gap = 4
    for idx, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = idx
            gap_start = None
        elif start is not None and gap_start is None:
            gap_start = idx
        elif (start is not None and gap_start is not None
              and idx - gap_start >= min_gap):
            if gap_start - start >= 28:
                ranges.append((start, gap_start))
            start = None
            gap_start = None
    if start is not None:
        end = (gap_start if gap_start is not None
               and h - gap_start >= min_gap else h)
        if end - start >= 28:
            ranges.append((start, end))

    if len(ranges) <= 1:
        return [(x1, y1, x2, y2)]
    # Keep only real card-height rows; tiny rule/text fragments are not
    # independent panels.
    boxes = [(x1, y1 + rs, x2, y1 + re)
             for rs, re in ranges if re - rs >= 36]
    if len(boxes) <= 1:
        return [(x1, y1, x2, y2)]
    return boxes


def _corner_fill_fraction(filled: np.ndarray, c_frac: float = 0.12) -> float:
    """Max foreground fraction across the four bbox-corner patches.

    A circle/ellipse leaves every bbox corner empty; a square fills all
    four. Taking the max over corners means one filled corner is enough
    to vote "square corner". Patch size is a fraction of the min side —
    0.12 keeps the corner metric monotonic in the rounding radius.
    """
    h, w = filled.shape[:2]
    c = max(2, int(round(min(h, w) * c_frac)))
    patches = (
        filled[:c, :c], filled[:c, w - c:],
        filled[h - c:, :c], filled[h - c:, w - c:],
    )
    return max(float(p.sum()) / float(c * c) for p in patches)


def classify_filled_shape(crop_bgr: np.ndarray):
    """Classify a solid geometric-primitive crop into a native PPT shape.

    ``crop_bgr`` must be the text-erased, children-inpainted bbox crop of
    a ``container`` / ``internal`` element. Returns
    ``(shape, fill_hex, line_hex, radius)`` when the crop is a clean
    solid ellipse / rounded rect / rect, so the element can be emitted
    as an editable native shape instead of a flattened PNG. Returns None
    for gradients, photos, multi-colour content, or unmatched
    silhouettes — those stay on the PNG path unchanged.
    """
    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop_bgr.astype(np.int16) - 255).max(axis=2)
    border = np.concatenate([
        crop_bgr[:2, :].reshape(-1, 3),
        crop_bgr[-2:, :].reshape(-1, 3),
        crop_bgr[:, :2].reshape(-1, 3),
        crop_bgr[:, -2:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0).astype(np.int16)
    fg = np.abs(crop_bgr.astype(np.int16) - bg[None, None]).max(axis=2) > 14
    if float(fg.mean()) < 0.30:
        # The shape touches every crop edge, so the border ring sampled
        # the shape's own colour. Fall back to white-diff segmentation.
        fg = (((gray < 250) & (diff_white > 6))
              | ((hsv[:, :, 1] > 10) & (diff_white > 4)))
        if float(fg.mean()) < 0.30:
            return None
    # Only clearly visible solids qualify: ghost-pale blocks (dashed
    # photo placeholders, faint tints barely off the panel colour) lose
    # their border detail as a flat native fill, so keep them as PNGs.
    gray_med = float(np.median(gray[fg]))
    sat_med = float(np.median(hsv[:, :, 1][fg]))
    if gray_med >= 235 and sat_med <= 25:
        return None
    # Fill interior holes (erased text leaves gaps inside badges) so the
    # coverage metrics describe the silhouette, not the gaps. A hole is
    # a background component NOT touching the crop border.
    inv = (~fg).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(inv, 4)
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
        | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    hole_mask = np.isin(
        labels, [i for i in range(1, n_labels) if i not in border_labels])
    filled = fg | hole_mask
    cov = float(filled.mean())
    corner = _corner_fill_fraction(filled)

    pixels = crop_bgr[fg].reshape(-1, 3)
    if len(pixels) < 24:
        return None
    quant = pixels.astype(np.uint16) // 16
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    key = keys[int(np.argmax(counts))]
    close = pixels[(quant == key).all(axis=1)]
    fill_bgr = np.median(close, axis=0).astype(np.int16)
    spread = np.abs(pixels.astype(np.int16) - fill_bgr[None, :]).max(axis=1)
    if float((spread <= 30).mean()) < 0.80:
        return None  # gradient / photo / multi-colour content
    # Interior must be uniform too: a glyph or photo baked inside the
    # shape would be lost by a native-shape emit, so keep the PNG.
    core = cv2.erode(filled.astype(np.uint8) * 255,
                     np.ones((3, 3), np.uint8), iterations=2) > 0
    if int(core.sum()) >= 24:
        core_pixels = crop_bgr[core].reshape(-1, 3)
        core_spread = np.abs(
            core_pixels.astype(np.int16) - fill_bgr[None, :]).max(axis=1)
        if float((core_spread <= 30).mean()) < 0.97:
            return None
    fg_u8 = fg.astype(np.uint8) * 255
    boundary = (fg_u8 > 0) & (cv2.erode(
        fg_u8, np.ones((3, 3), np.uint8), iterations=2) == 0)
    if int(boundary.sum()) >= 24:
        line_bgr = np.median(crop_bgr[boundary], axis=0).astype(np.int16)
    else:
        line_bgr = fill_bgr
    fill_hex = _bgr_to_hex(fill_bgr)
    line_hex = _bgr_to_hex(line_bgr)
    # Thresholds calibrated on rasterized squares with corner radius
    # r/s in [0, 0.5]: corner(c=0.12) falls monotonically 1.0 -> 0.0
    # while cov falls 1.0 -> 0.785 (circle). Mid gaps fall back to PNG.
    if cov >= 0.90 and corner >= 0.55:
        return ("rect", fill_hex, line_hex, 0.0)
    if cov >= 0.84 and corner <= 0.45:
        # Corner radius from the area lost to rounding:
        # 1 - cov = (4 - pi) * (r / side)^2 for a near-square shape.
        radius = math.sqrt(max(0.0, 1.0 - cov) / (4.0 - math.pi))
        radius = min(0.5, max(0.08, radius))
        return ("round_rect", fill_hex, line_hex, round(radius, 3))
    if 0.60 <= cov <= 0.835 and corner <= 0.03:
        return ("oval", fill_hex, line_hex, 0.0)
    return None


def classify_outline_ring(
        source: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask_path: str | None = None,
) -> tuple[str, float] | None:
    """Classify a near-square card-outline candidate.

    ``_outline_should_be_native_shape`` deliberately excludes near-square
    outlines: they are frequently circles, and a rounded rectangle drawn
    over a circular ring shows an extra visible box. This classifier
    recovers the safe half of those cases:
      * ring pixels hug one enclosing circle -> ("oval", 0.0)
      * ring runs along all four straight sides -> ("round_rect", radius)
    Anything else (concentric rings, polygons, content-polluted masks)
    returns None so the pixel-perfect PNG path is kept.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = source[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if min(h, w) < 40:
        return None
    mask = None
    if mask_path and Path(mask_path).exists():
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape[:2] == source.shape[:2]:
            mask = m[y1:y2, x1:x2] > 16
    if mask is None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        diff_white = np.abs(crop.astype(np.int16) - 255).max(axis=2)
        mask = (((gray < 253) & (diff_white > 5))
                | ((hsv[:, :, 1] > 8) & (diff_white > 4)))
    if float(mask.mean()) < 0.006:
        return None
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None:
        return None
    (_cx, _cy), r = cv2.minEnclosingCircle(pts)
    if r < 20:
        return None
    # Circle rings hug one circle: the radial distance distribution is a
    # narrow band. Calibrated on detector-thickness rings: circles measure
    # 0.06-0.14, rounded-square rings >=0.18 (r=0.3), squares ~0.35.
    d = np.sqrt((pts[:, 0, 0] - _cx) ** 2 + (pts[:, 0, 1] - _cy) ** 2)
    p05, p50, p95 = np.percentile(d, [5, 50, 95])
    if (p95 - p05) / max(1.0, p50) <= 0.16:
        return ("oval", 0.0)
    band = max(2, int(round(min(h, w) * 0.03)))
    def _straight(zone: np.ndarray, axis: int) -> float:
        # Fraction of columns (rows) whose band is at least half filled.
        # Circle tangents fill the band only near the centre column; a
        # straight card edge fills it across the whole run.
        fills = zone.mean(axis=axis)
        return float((fills >= 0.5).mean())
    sides = (
        _straight(mask[:band, :], 0),
        _straight(mask[-band:, :], 0),
        _straight(mask[:, :band], 1),
        _straight(mask[:, -band:], 1),
    )
    if min(sides) < 0.50:
        return None
    # Estimate the corner radius from the ring's closest approach to the
    # bbox corner: dist = r * (sqrt(2) - 1) for a rounded corner.
    c = max(3, int(round(min(h, w) * 0.25)))
    ys, xs = np.nonzero(mask[:c, :c])
    if len(xs):
        dist = float(np.min(np.hypot(xs, ys))) + 2.0
        radius = min(0.5, max(0.08, dist / 0.414 / min(h, w)))
    else:
        radius = 0.08
    return ("round_rect", round(radius, 3))
