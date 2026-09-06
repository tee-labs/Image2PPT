"""Outline / card shape helpers.

When a detected component looks like a card border (outline mask
exists), these helpers decide whether to:
  * emit it as a native PPT shape (rounded rectangle, oval, ...), or
  * keep the original alpha-masked image crop, or
  * split a tall outline into multiple stacked sub-cards.

The same module also hosts the solid-primitive classifier that lifts
small filled circles / rects (badges, colour chips) and thin circle
rings out of the PNG crop path into editable native shapes.
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


def _dominant_color(crop_bgr: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Median BGR of the dominant quantised colour inside ``fg``."""
    pixels = crop_bgr[fg].reshape(-1, 3)
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.int16)
    quant = pixels.astype(np.uint16) // 16
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    key = keys[int(np.argmax(counts))]
    close = pixels[(quant == key).all(axis=1)]
    if len(close) == 0:
        return np.median(pixels, axis=0).astype(np.int16)
    return np.median(close, axis=0).astype(np.int16)


def _ring_candidate(crop_bgr: np.ndarray, fg: np.ndarray, bg: np.ndarray):
    """Classify a sparse annulus silhouette (circle ring / donut).

    Covers thin circle outlines (badge rings, decorative circles) whose
    coverage is far below the filled-solid threshold. Returns
    ``("oval", None, line_hex, 0.0, thickness_px)`` — fill None means
    transparent — or None when the silhouette is not a single centered
    ring (multi-colour, off-centre hole, arcs, scattered fragments).
    """
    h, w = fg.shape[:2]
    if abs(w / float(h) - 1.0) > 0.33:
        return None
    fg_u8 = fg.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(fg_u8, 8)
    if n_labels - 1 == 0:
        return None
    counts = np.bincount(labels.ravel())[1:]
    # Anti-aliased rings often shatter into hairline fragments; require
    # one dominant component instead of a perfect single contour.
    if int(counts.max()) < 0.85 * int(fg.sum()):
        return None  # scattered fragments, not a ring
    # Bridge hairline gaps so the interior hole survives segmentation
    # noise; genuinely broken rings (arcs, notches) stay open and fail
    # the hole test below.
    closed = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    # Interior hole: background component not touching the crop border.
    inv = (~closed.astype(bool)).astype(np.uint8)
    n_inv, inv_labels = cv2.connectedComponents(inv, 4)
    border_labels = set(inv_labels[0, :].tolist()) \
        | set(inv_labels[-1, :].tolist()) \
        | set(inv_labels[:, 0].tolist()) \
        | set(inv_labels[:, -1].tolist())
    hole = np.isin(
        inv_labels, [i for i in range(1, n_inv) if i not in border_labels])
    hole_px = int(hole.sum())
    fg_px = int(fg.sum())
    if hole_px < max(80, int(0.15 * fg_px)):
        return None  # filled solid or C-shape
    ys, xs = np.nonzero(fg)
    hys, hxs = np.nonzero(hole)
    cx = float(xs.mean()); cy = float(ys.mean())
    if np.hypot(hxs.mean() - cx, hys.mean() - cy) > 0.18 * min(w, h):
        return None  # off-centre hole (C shapes, crescents)
    r = np.hypot(xs - cx, ys - cy)
    p05, p50, p95 = np.percentile(r, [5, 50, 95])
    if p50 < 8.0 or (p95 - p05) / max(1.0, p50) > 0.65:
        return None  # not a ring band (arcs, half-discs, blobs)
    # Square / rounded-square frames also produce a ring-like radial
    # band, but their mid-radius swings with the angle (side vs corner:
    # up to √2−1 ≈ 0.41). A true annulus keeps a near-constant mid
    # radius at every angle. Binning by angle catches the frames so
    # they fall through to the quad-ring classifier.
    ang = np.arctan2(ys - cy, xs - cx)
    bins = np.clip(((ang + np.pi) / (2 * np.pi) * 36).astype(int), 0, 35)
    med_r = np.zeros(36)
    for b in range(36):
        sel = bins == b
        if sel.any():
            med_r[b] = float(np.median(r[sel]))
    valid = med_r[med_r > 0]
    if (len(valid) >= 24
            and (valid.max() - valid.min())
            / max(1.0, float(np.median(valid))) > 0.18):
        return None  # straight-sided frame → quad path
    # Stroke colour: thin rings carry a wide anti-alias halo that blends
    # toward the background, so sample only the stroke CORE — the ring
    # pixels farthest from bg — and require the core itself to be one
    # uniform colour (gradient rings keep their PNG).
    px_all = crop_bgr[fg].reshape(-1, 3).astype(np.int16)
    dist_bg = np.abs(px_all - bg[None, :]).max(axis=1)
    core = px_all[dist_bg >= np.percentile(dist_bg, 60)]
    if len(core) < 12:
        return None
    line_bgr = np.median(core, axis=0).astype(np.int16)
    spread = np.abs(core - line_bgr[None, :]).max(axis=1)
    if float((spread <= 36).mean()) < 0.75:
        return None
    return ("oval", None, _bgr_to_hex(line_bgr), 0.0,
            float(np.percentile(r, 95) - np.percentile(r, 5)))


def _quad_ring_candidate(crop_bgr: np.ndarray, fg: np.ndarray,
                         bg: np.ndarray):
    """Classify a rectangular ring (box frame) silhouette.

    ``_ring_candidate`` only accepts near-square annuli — a wide hollow
    rectangle frame (aspect 2:1, 3:1 …) fails its aspect gate and the
    frame stays a flattened PNG. This parallel classifier covers the
    quad case: one dominant component, a centred interior hole, ink on
    all four straight sides, one uniform stroke colour. Returns
    ``("rect"|"round_rect", None, line_hex, radius, thickness_px)`` or
    None.
    """
    h, w = fg.shape[:2]
    fg_u8 = fg.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(fg_u8, 8)
    if n_labels - 1 == 0:
        return None
    counts = np.bincount(labels.ravel())[1:]
    if int(counts.max()) < 0.85 * int(fg.sum()):
        return None  # scattered fragments, not a frame
    closed = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    inv = (~closed.astype(bool)).astype(np.uint8)
    n_inv, inv_labels = cv2.connectedComponents(inv, 4)
    border_labels = set(inv_labels[0, :].tolist()) \
        | set(inv_labels[-1, :].tolist()) \
        | set(inv_labels[:, 0].tolist()) \
        | set(inv_labels[:, -1].tolist())
    hole = np.isin(
        inv_labels, [i for i in range(1, n_inv) if i not in border_labels])
    hole_px = int(hole.sum())
    fg_px = int(fg.sum())
    if hole_px < max(80, int(0.25 * fg_px)):
        return None  # filled solid, L-shape, or open frame
    ys, xs = np.nonzero(fg)
    hys, hxs = np.nonzero(hole)
    if np.hypot(hxs.mean() - xs.mean(), hys.mean() - ys.mean()) \
            > 0.20 * max(w, h):
        return None  # off-centre hole (not a simple frame)
    # Ink must hug all four bbox edges (straight sides). Measure the
    # per-column / per-row nearest-ink distance to the edge rather than
    # band fill: a 2-3 px stroke sitting exactly on the crop edge fills
    # only a third of a wider band.
    tol = max(2, min(h, w) // 24)
    top = np.argmax(fg_u8, axis=0)
    bot = np.argmax(fg_u8[::-1, :], axis=0)
    left = np.argmax(fg_u8, axis=1)
    right = np.argmax(fg_u8[:, ::-1], axis=1)
    col_any = fg_u8.any(axis=0)
    row_any = fg_u8.any(axis=1)
    f_top = float(((top <= tol) & col_any).sum()) / max(1, w)
    f_bot = float(((bot <= tol) & col_any).sum()) / max(1, w)
    f_left = float(((left <= tol) & row_any).sum()) / max(1, h)
    f_right = float(((right <= tol) & row_any).sum()) / max(1, h)
    if min(f_top, f_bot, f_left, f_right) < 0.65:
        return None  # curved or open silhouette, not a frame
    # One uniform stroke colour, sampled from the stroke core (farthest
    # from bg) so the anti-alias halo doesn't wash out the estimate.
    px_all = crop_bgr[fg].reshape(-1, 3).astype(np.int16)
    dist_bg = np.abs(px_all - bg[None, :]).max(axis=1)
    core = px_all[dist_bg >= np.percentile(dist_bg, 60)]
    if len(core) < 12:
        return None
    line_bgr = np.median(core, axis=0).astype(np.int16)
    spread = np.abs(core - line_bgr[None, :]).max(axis=1)
    if float((spread <= 36).mean()) < 0.75:
        return None
    # Corner shape from the stroke's closest approach to the bbox
    # corner: a sharp frame's stroke runs into the corner (dist ≈ 0),
    # a rounded corner keeps a stand-off of r·(√2−1)·side. Patch-fill
    # metrics are unreliable here — the hole leaks into the corner
    # patch through the inner arc.
    c = max(3, int(round(min(h, w) * 0.25)))
    corner_ys, corner_xs = np.nonzero(fg_u8[:c, :c] > 0)
    if len(corner_xs):
        dist = float(np.min(np.hypot(corner_xs, corner_ys)))
    else:
        dist = tol * 4.0
    if dist <= 2.5:
        kind, radius = "rect", 0.0
    else:
        radius = min(0.5, max(0.08, (dist + 2.0) / 0.414 / min(h, w)))
        kind, radius = "round_rect", round(radius, 3)
    # Stroke thickness ≈ fg px / approximate centerline length.
    thickness = fg_px / float(max(1.0, 2.0 * (w + h)))
    return (kind, None, _bgr_to_hex(line_bgr), radius,
            float(min(max(thickness, 1.0), 40.0)))


def classify_filled_shape(crop_bgr: np.ndarray):
    """Classify a solid geometric-primitive crop into a native PPT shape.

    ``crop_bgr`` must be the text-erased, children-inpainted bbox crop of
    an image element. Returns
    ``(shape, fill_hex, line_hex, radius, line_px)`` when the crop is a
    clean solid ellipse / rounded rect / rect (``line_px`` 0.0; the
    builder keeps its fixed border width), or a ring annulus
    (``fill_hex`` None → transparent, ``line_px`` ring thickness in
    source pixels), so the element can be emitted as an editable native
    shape instead of a flattened PNG. Returns None for gradients,
    photos, multi-colour content, or unmatched silhouettes — those stay
    on the PNG path unchanged.
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
            # Sparse silhouettes (thin rings) can't reach the filled
            # branches, but the ring branches may still claim them.
            if float(fg.mean()) < 0.02:
                return None
            ring = _ring_candidate(crop_bgr, fg, bg)
            if ring is not None:
                return ring
            return _quad_ring_candidate(crop_bgr, fg, bg)
    elif float(fg.mean()) > 0.60:
        # Hollow frame touching the crop edge: the border ring sampled
        # the frame's stroke, inverting the mask (interior reads as
        # foreground). Re-segment against the white reference and keep
        # the result when it actually separates a sparse silhouette.
        cand = (((gray < 250) & (diff_white > 6))
                | ((hsv[:, :, 1] > 10) & (diff_white > 4)))
        if float(cand.mean()) < float(fg.mean()):
            fg = cand
    if float(fg.mean()) < 0.55:
        ring = _ring_candidate(crop_bgr, fg, bg)
        if ring is not None:
            return ring
        quad = _quad_ring_candidate(crop_bgr, fg, bg)
        if quad is not None:
            return quad
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
    fill_bgr = _dominant_color(crop_bgr, fg)
    spread = np.abs(pixels.astype(np.int16) - fill_bgr[None, :]).max(axis=1)
    if float((spread <= 30).mean()) < 0.80:
        return None  # gradient / photo / multi-colour content
    # Interior must be uniform too: a glyph or photo baked inside the
    # shape would be lost by a native-shape emit, so keep the PNG. The
    # erosion must clear the shape's own border stroke (2-3 px + anti-
    # alias), not just 2 iterations — border remnants read as interior
    # spread and rejected pale bordered cards at 0.96 vs the 0.97 gate.
    # borderValue=0 matters: cv2's erode default pads with +inf, which
    # lets the border stroke survive on crop-edge rows of tight bboxes
    # and pollute the core.
    core_depth = max(3, min(h, w) // 24)
    core = cv2.erode(filled.astype(np.uint8) * 255,
                     np.ones((3, 3), np.uint8),
                     iterations=core_depth,
                     borderType=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0
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
    # Ghost-pale fill gate (after colour sampling): faint tint blocks
    # stay PNG unless a crisp solid border exists — pale cards with a
    # visible outline are the classic deck card and belong in the
    # native-shape path. Border evidence comes from either the
    # silhouette boundary (padded crop: the border ring around the crop
    # sampled white so the stroke lands on the boundary) or the
    # segmentation bg itself (tight crop: the border ring IS the card's
    # stroke). Dashed placeholders and borderless tint panels have
    # neither → PNG.
    gray_med = float(np.median(gray[fg]))
    sat_med = float(np.median(hsv[:, :, 1][fg]))
    if gray_med >= 235 and sat_med <= 25:
        def _border_like(colour: np.ndarray) -> bool:
            diff = float(np.max(
                np.abs(colour.astype(np.int16) - fill_bgr.astype(np.int16))))
            lum = float(0.114 * colour[0]
                        + 0.587 * colour[1] + 0.299 * colour[2])
            return diff > 30 and lum < 235
        if _border_like(bg) and not _border_like(line_bgr):
            line_bgr = bg.copy()
            line_hex = _bgr_to_hex(line_bgr)
        if not _border_like(line_bgr):
            return None
    # Thresholds calibrated on rasterized squares with corner radius
    # r/s in [0, 0.5]: corner(c=0.12) falls monotonically 1.0 -> 0.0
    # while cov falls 1.0 -> 0.785 (circle). Mid gaps fall back to PNG.
    if cov >= 0.90 and corner >= 0.55:
        return ("rect", fill_hex, line_hex, 0.0, 0.0)
    if cov >= 0.84 and corner <= 0.45:
        # Corner radius from the area lost to rounding:
        # 1 - cov = (4 - pi) * (r / side)^2 for a near-square shape.
        radius = math.sqrt(max(0.0, 1.0 - cov) / (4.0 - math.pi))
        radius = min(0.5, max(0.08, radius))
        return ("round_rect", fill_hex, line_hex, round(radius, 3), 0.0)
    if 0.60 <= cov <= 0.835 and corner <= 0.03:
        return ("oval", fill_hex, line_hex, 0.0, 0.0)
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
