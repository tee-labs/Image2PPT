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
    all four straight sides, one uniform stroke colour. A DASHED frame
    shatters into many dash components; a dash-scale morphological
    close is tried before giving up, and the kind gains a ``dashed_``
    prefix so the PPTX builder restores the dash style. Returns
    ``("rect"|"round_rect"|"dashed_rect"|"dashed_round_rect", None,
    line_hex, radius, thickness_px)`` or None.
    """
    h, w = fg.shape[:2]
    fg_u8 = fg.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(fg_u8, 8)
    if n_labels - 1 == 0:
        return None
    counts = np.bincount(labels.ravel())[1:]
    orig_fg_px = int(fg.sum())
    colour_fg = fg
    dashed = False
    if int(counts.max()) < 0.85 * orig_fg_px:
        # Scattered fragments — but a dashed frame's dashes are exactly
        # that. Bridge dash gaps ALONG each stroke with directional
        # closes (a square close cannot seal the diagonal corner crack
        # between two dash-phase-offset strokes), then stamp the four
        # corner blocks so the ring encloses the interior hole.
        work = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE,
                                np.ones((1, 11), np.uint8))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE,
                                np.ones((11, 1), np.uint8))
        gap = 14
        work[:gap, :gap] = 255
        work[:gap, -gap:] = 255
        work[-gap:, :gap] = 255
        work[-gap:, -gap:] = 255
        n2, labels2 = cv2.connectedComponents(work, 8)
        if n2 - 1 == 0:
            return None
        counts2 = np.bincount(labels2.ravel())[1:]
        if int(counts2.max()) < 0.85 * int((work > 0).sum()):
            return None
        fg_u8 = work
        fg = work.astype(bool)
        dashed = True
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
    ys, xs = np.nonzero(fg)
    if hole_px < max(80, int(0.25 * fg_px)):
        return None  # filled solid, L-shape, or open frame
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
    # from bg) so the anti-alias halo doesn't wash out the estimate. For
    # a dashed frame sample the ORIGINAL dash ink — the bridged mask is
    # mostly gap pixels whose image colour is the background.
    px_all = crop_bgr[colour_fg].reshape(-1, 3).astype(np.int16)
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
    if dashed:
        kind = f"dashed_{kind}"
    # Stroke thickness ≈ fg px / approximate centerline length. For a
    # dashed frame the bridged mask would inflate the estimate, so use
    # the original dash ink.
    thickness = (orig_fg_px if dashed else fg_px) \
        / float(max(1.0, 2.0 * (w + h)))
    return (kind, None, _bgr_to_hex(line_bgr), radius,
            float(min(max(thickness, 1.0), 40.0)))


def _arrow_axis_profile(filled: np.ndarray):
    """Score an oriented (right-pointing) block-arrow silhouette.

    Returns ``(shaft_px, head_len_px)`` when the per-column ink-height
    profile shows a constant-thickness stem on the left and a
    full-height triangular head tapering to a tip on the right, else
    None. Works on the hole-filled, tight-bbox-trimmed silhouette.
    """
    h, w = filled.shape
    if min(h, w) < 24:
        return None
    col_h = filled.sum(axis=0).astype(np.int32)
    full = np.nonzero(col_h >= 0.85 * h)[0]
    if len(full) == 0:
        return None
    head_start = int(full[0])
    if head_start < max(4, 0.15 * w) or head_start > 0.85 * w:
        return None
    stem_zone = col_h[:head_start]
    if len(stem_zone) < 6:
        return None
    shaft = float(np.median(stem_zone[len(stem_zone) // 8:]))
    if shaft < 3 or shaft > 0.75 * h:
        return None
    if float(np.max(np.abs(stem_zone - shaft))) > max(2.0, 0.30 * shaft):
        return None
    # The stem must be ONE solid horizontal band (a flag/banner pair of
    # bars leaves background rows between them).
    stem_rows = np.nonzero(filled[:, :head_start].any(axis=1))[0]
    if len(stem_rows) == 0:
        return None
    if stem_rows[-1] - stem_rows[0] + 1 > shaft + max(3.0, 0.15 * shaft):
        return None
    # Head tapers to a tip: the last column must be nearly empty.
    if col_h[-1] > 0.35 * h:
        return None
    head_len = w - head_start
    if head_len < 0.10 * w:
        return None
    return shaft, float(head_len)


def _arrow_candidate(filled: np.ndarray):
    """Lift a solid block arrow (MSO right/left/up/down arrow).

    The oriented profile test runs on four re-orientations of the
    silhouette; the winning orientation names the MSO autoshape so no
    rotation is needed. Returns ``(kind, adj1, adj2)`` — shaft-thickness
    and head-length adjustments in the preset's own frame — or None.
    """
    variants = (
        ("right_arrow", filled),
        ("left_arrow", filled[:, ::-1]),
        ("down_arrow", filled.T),
        ("up_arrow", filled.T[:, ::-1]),
    )
    for kind, mask in variants:
        mask = np.ascontiguousarray(mask)
        hit = _arrow_axis_profile(mask)
        if hit is None:
            continue
        shaft, head_len = hit
        h_o, w_o = mask.shape
        ss = max(1.0, float(min(h_o, w_o)))
        adj1 = min(1.0, max(0.05, shaft / float(h_o)))
        adj2 = min(w_o / ss, max(0.05, head_len / ss))
        return kind, round(adj1, 3), round(adj2, 3)
    return None


def arrow_geometry(crop_bgr: np.ndarray):
    """Re-measure block-arrow adjustments for a classified arrow crop.

    ``classify_filled_shape`` keeps its 5-tuple return contract, so the
    builder calls this once on the same crop when the classified kind is
    an arrow. Returns ``(kind, [adj1, adj2])`` or None.
    """
    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return None
    border = np.concatenate([
        crop_bgr[:2, :].reshape(-1, 3),
        crop_bgr[-2:, :].reshape(-1, 3),
        crop_bgr[:, :2].reshape(-1, 3),
        crop_bgr[:, -2:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0).astype(np.int16)
    fg = np.abs(crop_bgr.astype(np.int16) - bg[None, None]).max(axis=2) > 14
    if float(fg.mean()) < 0.30:
        return None
    inv = (~fg).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(inv, 4)
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
        | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    hole_mask = np.isin(
        labels, [i for i in range(1, n_labels) if i not in border_labels])
    filled = fg | hole_mask
    ys, xs = np.nonzero(filled)
    if len(xs) == 0:
        return None
    filled = filled[int(ys.min()):int(ys.max()) + 1,
                    int(xs.min()):int(xs.max()) + 1]
    return _arrow_candidate(filled)


def _banner_candidate(ap: np.ndarray, w: int, h: int) -> str | None:
    """Right-pointing process banners: homeplate (flat left edge) and
    chevron (notched left edge).

    MSO draws both pointing right, so only that orientation lifts.
    Slot structure from the stable polygon approximation:
      homeplate — 5 vertices: top pair (0, xa), bottom pair (0, xa),
                  apex at (w, mid).
      chevron   — 6 vertices: same four shoulders plus a reflex notch
                  vertex at (xn, mid) on the left edge.
    """
    tol = max(3.0, 0.12 * min(w, h))
    if h < 20 or w < 30:
        return None
    top = [p for p in ap if p[1] <= tol]
    bot = [p for p in ap if p[1] >= h - 1 - tol]
    right = [p for p in ap
             if p[0] >= w - 1 - tol and abs(p[1] - (h - 1) / 2.0) <= tol]
    mid = [p for p in ap if tol < p[1] < h - 1 - tol]
    if len(top) != 2 or len(bot) != 2:
        return None
    if not all(abs(p[1] - (h - 1) / 2.0) <= tol for p in mid):
        return None
    tx = sorted(float(p[0]) for p in top)
    bx = sorted(float(p[0]) for p in bot)
    if abs(tx[0] - bx[0]) > tol or abs(tx[1] - bx[1]) > tol:
        return None  # shoulders must align vertically
    if tx[0] > 2.0 * tol:
        return None  # left edge not flush with the bbox side
    if tx[1] < 0.25 * (w - 1) or tx[1] > 0.92 * (w - 1):
        return None  # shoulder must sit between flush and apex
    if len(mid) == 1 and len(right) == 1 \
            and abs(mid[0][0] - right[0][0]) <= 1.0:
        return "homeplate"
    if len(mid) == 2 and len(right) == 1:
        notch = [p for p in mid if p[0] < tx[1] - 0.02 * w]
        apex = [p for p in mid if abs(p[0] - (w - 1)) <= tol]
        if len(apex) == 1 and len(notch) == 1 \
                and tx[0] <= notch[0][0] <= tx[1]:
            return "chevron"
    return None


def _regular_polygon_slots(pts: np.ndarray, n: int,
                           w: int, h: int) -> bool:
    """Orientation gate for regular pentagon / hexagon lifts.

    MSO draws REGULAR_PENTAGON point-up (one apex top-centre, a
    horizontal base on the bbox bottom) and HEXAGON flat-top with
    points left+right. Only silhouettes matching those orientations
    lift — a rotated polygon would render at the wrong angle.
    Side-length uniformity is checked by the caller's approx stability
    plus this helper's near-equal-side gate.
    """
    sides = sorted(
        float(math.hypot(*(pts[(i + 1) % n] - pts[i]))) for i in range(n))
    if sides[-1] > 1.30 * max(1.0, sides[0]):
        return False
    if n == 5:
        apex = [p for p in pts
                if p[1] <= 0.16 * (h - 1)
                and abs(p[0] - (w - 1) / 2.0) <= 0.22 * w]
        if len(apex) != 1:
            return False
        base = [p for p in pts if p[1] >= 0.90 * (h - 1)]
        if len(base) != 2 or abs(base[0][1] - base[1][1]) > 0.08 * h:
            return False
        return abs(base[0][0] - base[1][0]) >= 0.45 * w
    top = [p for p in pts if p[1] <= 0.12 * (h - 1)]
    bot = [p for p in pts if p[1] >= 0.88 * (h - 1)]
    left = [p for p in pts if p[0] <= 0.12 * (w - 1)]
    right = [p for p in pts if p[0] >= 0.88 * (w - 1)]
    return (len(top) == 2 and len(bot) == 2
            and len(left) == 1 and len(right) == 1)


def _polygon_candidate(filled: np.ndarray) -> str | None:
    """Convex-polygon lift for silhouettes the rect/oval gates rejected.

    Runs after the colour-uniformity gates, so only geometry decides.
    Upright primitives only — kinds map straight onto MSO autoshapes
    with no rotation:
      * 3 stable vertices, horizontal base on a bbox edge -> "triangle"
      * 4 stable vertices at bbox-edge midpoints          -> "diamond"
      * 4 stable vertices, two horizontal parallel edges,
        shorter one on top (MSO trapezoid orientation)    -> "trapezoid"
    Everything else (rotated quads, pentagons, concave glyphs, blobs)
    returns None so the crop stays on the pixel-perfect PNG path.
    """
    h, w = filled.shape[:2]
    if min(h, w) < 16:
        return None
    cnts, _ = cv2.findContours(
        filled.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.90 * float(filled.sum()):
        return None  # frilly silhouette (glyph/icon), not a clean polygon
    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None
    approxes = [cv2.approxPolyDP(cnt, eps * peri, True)
                for eps in (0.022, 0.035)]
    if len({len(a) for a in approxes}) != 1:
        return None  # vertex count unstable across epsilons -> blob
    approx = approxes[0]
    # Process banners (homeplate / chevron) are slot-checked before the
    # vertex-sharpness gate: a shallow head meets the horizontal edge at
    # ~35-40°, below the arc-rejection threshold, and the banner slot
    # structure (aligned shoulders + centred apex/notch) is itself a
    # strong gate against arc-collapse false positives.
    ap_early = approx.reshape(-1, 2).astype(np.float64)
    banner = _banner_candidate(ap_early, w, h)
    if banner is not None:
        return banner
    # Sharpness: a rounded rect's arc contour collapses to a slanted
    # quad at these epsilons, and its arc perimeter far exceeds the
    # chord polygon's. A true polygon keeps the two within a few
    # percent.
    if cv2.arcLength(approx, True) < 0.92 * peri:
        return None
    # Vertex sharpness, snap-first: approxPolyDP may slide a vertex
    # along an edge or cut a convex corner by up to eps, which corrupts
    # any local angle measurement. Convex silhouettes let us recover
    # the true corner: snap each vertex to the contour point extreme
    # along its outward bisector (the support point), then measure the
    # contour turn there. A true polygon vertex turns hard (>= 60° for
    # hexagon..square); a collapsed corner arc of radius r turns only
    # ~90°·(2·win)/(pi·r/2) — a 5px window on r >= 20 stays under 30°.
    pts_seq = cnt.reshape(-1, 2).astype(np.float64)
    m = len(pts_seq)
    centroid = pts_seq.mean(axis=0)
    ap = approx.reshape(-1, 2).astype(np.float64)
    win = max(3, int(round(0.05 * min(h, w))))
    snapped = []
    for i, v in enumerate(ap):
        v_prev, v_next = ap[(i - 1) % len(ap)], ap[(i + 1) % len(ap)]
        t1 = v_prev - v
        t2 = v_next - v
        n1v = float(np.hypot(*t1))
        n2v = float(np.hypot(*t2))
        if n1v < 1.0 or n2v < 1.0:
            snapped.append(None)
            continue
        # Both vectors point from v along the polygon edges: their sum
        # is the INTERIOR bisector for a convex corner, so negate for
        # outward; the centroid dot-check guards orientation slips.
        bis = -(t1 / n1v + t2 / n2v)
        nb = float(np.hypot(*bis))
        if nb < 1e-6:
            snapped.append(None)
            continue
        bis /= nb
        if np.dot(bis, v - centroid) < 0:
            bis = -bis
        snapped.append(int(np.argmax(pts_seq @ bis)))
    for v, idx in zip(ap, snapped):
        if idx is None:
            continue
        a = pts_seq[(idx - win) % m]
        b = pts_seq[(idx + win) % m]
        vv = pts_seq[idx]
        d1, d2 = vv - a, b - vv
        n1 = float(math.hypot(d1[0], d1[1]))
        n2 = float(math.hypot(d2[0], d2[1]))
        if n1 < 1.0 or n2 < 1.0:
            continue
        cosang = float(np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0))
        if math.degrees(math.acos(cosang)) < 40.0:
            return None  # collapsed arc, not a sharp polygon vertex
    approx = approxes[0]
    if not cv2.isContourConvex(approx):
        return None
    pts = approx.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    if n == 3:
        # Upright triangle: longest edge near-horizontal AND sitting on
        # the bbox top or bottom edge (apex points the other way).
        e0, e1 = max(
            ((pts[i], pts[(i + 1) % n]) for i in range(n)),
            key=lambda p: float(np.hypot(*(p[1] - p[0]))),
        )
        length = float(np.hypot(*(e1 - e0)))
        if length >= 0.5 * max(w, h) \
                and abs(e1[1] - e0[1]) <= 0.22 * length \
                and (max(e0[1], e1[1]) >= 0.94 * (h - 1)
                     or min(e0[1], e1[1]) <= 0.06 * (h - 1)):
            return "triangle"
        return None
    if n in (5, 6) and _regular_polygon_slots(pts, n, w, h):
        return "pentagon" if n == 5 else "hexagon"
    if n != 4:
        return None
    horiz = 0
    for i in range(4):
        ex, ey = pts[(i + 1) % 4] - pts[i]
        if abs(ey) <= 0.20 * max(1.0, math.hypot(ex, ey)):
            horiz += 1
    if horiz == 2:
        # Two horizontal edges: trapezoid only when the widths differ
        # and the short one is on top (MSO default). Equal widths is a
        # parallelogram/rect — already judged by the rect gates.
        top_pair = sorted(pts, key=lambda p: p[1])[:2]
        bot_pair = sorted(pts, key=lambda p: p[1])[-2:]
        top_w = abs(top_pair[0][0] - top_pair[1][0])
        bot_w = abs(bot_pair[0][0] - bot_pair[1][0])
        if min(top_w, bot_w) >= 0.30 * w \
                and abs(top_w - bot_w) >= 0.25 * max(top_w, bot_w):
            # Short side on top is the MSO trapezoid default; a short
            # BOTTOM lifts as "trapezoid_down" (the PPTX builder adds a
            # 180° rotation) instead of staying a PNG.
            return "trapezoid" if top_w < bot_w else "trapezoid_down"
        # Equal-width horizontal edges that are horizontally offset:
        # a parallelogram. A rect's top/bottom mids align (slant ~ 0)
        # and fall through to the upright rect coverage band instead.
        # MSO's parallelogram has one fixed slant adjustment, so gate
        # to moderate offsets where the default autoshape stays
        # visually faithful.
        if min(top_w, bot_w) >= 0.30 * w \
                and abs(top_w - bot_w) <= 0.20 * max(top_w, bot_w):
            slant = abs(
                (top_pair[0][0] + top_pair[1][0]) / 2.0
                - (bot_pair[0][0] + bot_pair[1][0]) / 2.0)
            if 0.15 * min(w, h) <= slant <= 0.55 * w:
                return "parallelogram"
        return None
    if horiz >= 1:
        return None  # mixed quad (kite/house shapes): keep PNG
    # Diamond: each vertex near ITS OWN bbox-edge midpoint (a slanted
    # parallelogram shifts its top/bottom vertices off-centre and fails
    # the slot match). Covers the 45° rotated square.
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    tol = max(3.0, 0.12 * min(w, h))
    slots = {"top": False, "bottom": False, "left": False, "right": False}
    for x, y in pts:
        if abs(x - cx) <= tol and y <= 0.15 * h:
            slots["top"] = True
        elif abs(x - cx) <= tol and y >= 0.85 * h:
            slots["bottom"] = True
        elif abs(y - cy) <= tol and x <= 0.15 * w:
            slots["left"] = True
        elif abs(y - cy) <= tol and x >= 0.85 * w:
            slots["right"] = True
    if all(slots.values()):
        return "diamond"
    return None


def _ellipse_band_fit(mask: np.ndarray):
    """Fit an ellipse to silhouette ink; report how tightly the ink hugs
    the fitted perimeter.

    Returns ``None`` when the ink is too sparse, the band is not thin
    (blobs, concentric rings, frilly glyphs), or the ring is open (C
    shapes leave empty angular sectors). Otherwise returns
    ``(rho_p05, rho_p50, rho_p95, thickness_px)`` where ``rho`` is the
    normalised ellipse radius (1.0 == exactly on the perimeter).
    """
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None or len(pts) < 40:
        return None
    (ecx, ecy), (ea, eb), eang = cv2.fitEllipse(pts)
    a, b = max(ea, eb) / 2.0, min(ea, eb) / 2.0
    if a < 12.0 or b < 6.0:
        return None
    theta = math.radians(eang)
    xs = pts[:, 0, 0].astype(np.float64) - ecx
    ys = pts[:, 0, 1].astype(np.float64) - ecy
    u = xs * math.cos(theta) + ys * math.sin(theta)
    v = -xs * math.sin(theta) + ys * math.cos(theta)
    # ea/eb are FULL axis lengths, so a perimeter point measures rho=1.
    rho = 2.0 * np.sqrt((u / ea) ** 2 + (v / eb) ** 2)
    per = math.pi * (3.0 * (a + b)
                     - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
    thickness = float(mask.sum()) / max(1.0, per)
    # Thin closed band only: thick ink (solid diamonds, stars, blobs)
    # blows up the thickness-scaled tolerance and would classify as an
    # "oval ring" it is not. The absolute 0.45 cap catches multi-ring
    # silhouettes (concentric circles): two strokes double the
    # thickness estimate, but their rho spread stays wide.
    if thickness > 0.25 * b:
        return None
    tol = min(max(0.14, 3.0 * thickness / b), 0.45)
    p05, p50, p95 = np.percentile(rho, [5, 50, 95])
    if (p95 - p05) > tol:
        return None  # band too thick / scattered ink
    if abs(p50 - 1.0) > 0.75 * tol:
        return None  # ink not centred on the fitted perimeter
    ang_pt = np.arctan2(v, u)
    bins = np.clip(((ang_pt + np.pi) / (2 * np.pi) * 12).astype(int), 0, 11)
    if len(np.unique(bins)) < 10:
        return None  # open arc / C shape
    return p05, p50, p95, thickness


def _ellipse_ring_candidate(crop_bgr: np.ndarray, fg: np.ndarray,
                            bg: np.ndarray):
    """Classify a sparse wide-ellipse outline (decorative title halo).

    ``_ring_candidate`` only accepts near-square annuli and
    ``_quad_ring_candidate`` requires straight sides, so a wide thin
    ellipse ring fell through to the PNG path. Requires one dominant
    component hugging one fitted ellipse and a single uniform stroke
    colour. Returns ``("oval", None, line_hex, 0.0, thickness_px)`` or
    None.
    """
    h, w = fg.shape[:2]
    if min(h, w) < 24:
        return None
    fit = _ellipse_band_fit(fg)
    if fit is None:
        return None
    fg_u8 = fg.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(fg_u8, 8)
    if n_labels - 1 == 0:
        return None
    counts = np.bincount(labels.ravel())[1:]
    if int(counts.max()) < 0.85 * int(fg.sum()):
        return None  # scattered fragments, not one ring
    # Interior hole (4-connectivity, same as _ring_candidate): a closed
    # ring traps a background component; hairline 1-px strokes leak on
    # diagonal steps and stay on the PNG path like the circle rings do.
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
    if int(hole.sum()) < max(80, int(0.15 * int(fg.sum()))):
        return None  # open arc / hairline stroke
    px_all = crop_bgr[fg].reshape(-1, 3).astype(np.int16)
    dist_bg = np.abs(px_all - bg[None, :]).max(axis=1)
    core = px_all[dist_bg >= np.percentile(dist_bg, 60)]
    if len(core) < 12:
        return None
    line_bgr = np.median(core, axis=0).astype(np.int16)
    spread = np.abs(core - line_bgr[None, :]).max(axis=1)
    if float((spread <= 36).mean()) < 0.75:
        return None
    _p05, _p50, _p95, thickness = fit
    return ("oval", None, _bgr_to_hex(line_bgr), 0.0,
            float(max(1.0, thickness)))


def _outline_hugs_ellipse(mask: np.ndarray) -> bool:
    """True when outline ink forms one thin closed band around a fitted
    ellipse (a wide/skewed oval ring that the circle-hug test misses)."""
    return _ellipse_band_fit(mask) is not None


def _vet_fill_colors(crop_bgr: np.ndarray, fg: np.ndarray,
                     filled: np.ndarray, bg: np.ndarray):
    """Shared fill/line colour vetting for solid-primitive lifts.

    Accepts only one uniform ink colour with a clean interior: gradient
    or multi-colour content, glyphs/photos baked inside the shape, and
    ghost-pale fills without a crisp border all return None so the crop
    stays on the pixel-perfect PNG path. Returns
    ``(fill_hex, line_hex)``.
    """
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
    h, w = filled.shape[:2]
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
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
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
    return fill_hex, line_hex


def _rotated_rect_candidate(crop_bgr: np.ndarray):
    """Lift a solid rectangle drawn at a skew angle to a native rect.

    Upright primitives land in ``classify_filled_shape``; a rotated one
    (diagonal banner, tilted card, arrow tick) fills its bbox corners
    only partially and fails every upright coverage band, so it used to
    stay a flattened PNG. Returns
    ``(x, y, w, h, rotation_deg, fill_hex, line_hex)`` in crop
    coordinates — the box is the min-area rect because PPT rotates
    around the shape centre — or None. Tight gates: clearly skewed
    (|angle| >= 12°), near-perfect rectangular silhouette, one uniform
    fill colour.
    """
    h, w = crop_bgr.shape[:2]
    if min(h, w) < 24:
        return None
    border = np.concatenate([
        crop_bgr[:2, :].reshape(-1, 3),
        crop_bgr[-2:, :].reshape(-1, 3),
        crop_bgr[:, :2].reshape(-1, 3),
        crop_bgr[:, -2:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0).astype(np.int16)
    fg = np.abs(crop_bgr.astype(np.int16) - bg[None, None]).max(axis=2) > 14
    if float(fg.mean()) < 0.30:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        diff_white = np.abs(crop_bgr.astype(np.int16) - 255).max(axis=2)
        fg = (((gray < 250) & (diff_white > 6))
              | ((hsv[:, :, 1] > 10) & (diff_white > 4)))
        if float(fg.mean()) < 0.30:
            return None
    # Fill interior holes (erased text inside the shape) so both the
    # contour and the IoU metric describe the solid silhouette.
    inv = (~fg).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(inv, 4)
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
        | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    hole_mask = np.isin(
        labels, [i for i in range(1, n_labels) if i not in border_labels])
    filled = fg | hole_mask
    cnts, _ = cv2.findContours(
        filled.astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.92 * float(filled.sum()):
        return None  # frilly silhouette (icon/glyph), not one solid
    (rcx, rcy), (r1, r2), _ang = cv2.minAreaRect(cnt)
    s_long, s_short = max(r1, r2), min(r1, r2)
    if s_short < 14:
        return None
    box = cv2.boxPoints(((rcx, rcy), (r1, r2), _ang))
    poly = np.zeros(filled.shape, np.uint8)
    cv2.fillPoly(poly, [np.round(box).astype(np.int32)], 1)
    # PPT rotation is clockwise-positive around the shape centre; image
    # coords are y-down so atan2(dy, dx) of the long edge is already
    # clockwise-positive. Fold into (-90, 90], then express as
    # width=long-side rotation in [-45, 45).
    edges = sorted(
        ((float(np.hypot(box[(i + 1) % 4][0] - box[i][0],
                         box[(i + 1) % 4][1] - box[i][1])), i)
         for i in range(4)), reverse=True)
    p, q = box[edges[0][1]], box[(edges[0][1] + 1) % 4]
    angle = math.degrees(
        math.atan2(float(q[1] - p[1]), float(q[0] - p[0])))
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    if abs(angle) > 45.0:
        # Long edge runs near-vertical: make the short edge the PPT
        # width and fold the rotation by 90°.
        angle -= math.copysign(90.0, angle)
        s_long, s_short = s_short, s_long
    inter = int((poly.astype(bool) & filled).sum())
    union = int((poly.astype(bool) | filled).sum())
    # A true rasterized rotated rect measures ~0.97; a filled circle's
    # tilted min-area rect still reaches ~0.95, so IoU alone cannot
    # separate them. A rect has four sharp corners: warp the mask into
    # the rect frame and require filled corners (a circle inscribed in
    # its tilted square leaves every corner empty).
    if union == 0 or inter / float(union) < 0.93:
        return None  # ellipse / rounded blob — not rect enough
    th = math.radians(angle)
    ct, st = math.cos(th), math.sin(th)
    cw, chh = max(2, int(round(s_long))), max(2, int(round(s_short)))
    M = np.array([
        [ct, st, cw / 2.0 - ct * rcx - st * rcy],
        [-st, ct, chh / 2.0 + st * rcx - ct * rcy],
    ], np.float64)
    warped = cv2.warpAffine(
        filled.astype(np.uint8) * 255, M, (cw, chh),
        flags=cv2.INTER_NEAREST, borderValue=0) > 0
    if _corner_fill_fraction(warped) < 0.45:
        return None  # circle / oval — corners never fill
    if abs(angle) < 12.0:
        return None  # near-axis: the upright bands classify it better
    colors = _vet_fill_colors(crop_bgr, fg, filled, bg)
    if colors is None:
        return None
    fill_hex, line_hex = colors
    return (int(round(rcx - s_long / 2.0)), int(round(rcy - s_short / 2.0)),
            max(1, int(round(s_long))), max(1, int(round(s_short))),
            round(float(angle), 1), fill_hex, line_hex)


def classify_connector_line(crop_bgr: np.ndarray):
    """Lift a straight connector / divider stroke to a native line.

    The stroke must be one thin, straight, uniform-colour run of ink:
    PCA over the foreground gives the axis; a perp-spread gate rejects
    elbow polylines and scribbles; a per-segment width profile detects
    an arrowhead at one end; 1-px occupancy runs along the axis detect
    a dash / dot pattern. Returns
    ``(points, line_hex, width_px, dash, arrow)`` — points in crop
    coordinates, ``dash`` in {"dash", "dot", None}, ``arrow`` in
    {"start", "end", None} — or None so the crop stays on the PNG path.
    """
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return None
    border = np.concatenate([
        crop_bgr[:2, :].reshape(-1, 3),
        crop_bgr[-2:, :].reshape(-1, 3),
        crop_bgr[:, :2].reshape(-1, 3),
        crop_bgr[:, -2:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0).astype(np.int16)
    fg = np.abs(crop_bgr.astype(np.int16) - bg[None, None]).max(axis=2) > 14
    fg_frac = float(fg.mean())
    if not 0.002 <= fg_frac <= 0.55:
        return None
    ys, xs = np.nonzero(fg)
    if len(xs) < 24:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    cov = (pts - mean).T @ (pts - mean) / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    v1 = vecs[:, -1]  # major axis
    v2 = vecs[:, 0]
    proj = (pts - mean) @ v1
    perp = (pts - mean) @ v2
    length = float(proj.max() - proj.min())
    width_est = len(pts) / max(1.0, length)
    if length < 24 or width_est > max(6.0, 0.20 * length):
        return None
    p05, p95 = np.percentile(perp, [4, 96])
    if (p95 - p05) > max(5.0, 2.2 * width_est + 2.0):
        return None  # elbow polyline / L-shape / blob
    # Arrowhead: perp width per axial segment; a single-end bulge over
    # the last/first quarter marks a head. Both ends bulging is not a
    # simple arrow — keep the PNG.
    k = 8
    edges = np.linspace(proj.min(), proj.max(), k + 1)
    seg_w = []
    for i in range(k):
        sel = (proj >= edges[i]) & (proj <= edges[i + 1])
        if int(sel.sum()) < 3:
            seg_w.append(0.0)
            continue
        w05, w95 = np.percentile(perp[sel], [4, 96])
        seg_w.append(float(w95 - w05))
    mid = sorted(seg_w[2:6])
    mid_w = mid[len(mid) // 2] if mid else 0.0
    arrow = None
    head_min = max(6.0, 2.5 * mid_w)
    if seg_w[0] > head_min and seg_w[-1] > head_min:
        arrow = "both"
    elif seg_w[-1] > head_min:
        arrow = "end"
    elif seg_w[0] > head_min:
        arrow = "start"
    # Dash / dot: occupancy runs along the axis in 1-px bins.
    span = int(round(length)) + 1
    occ = np.zeros(span, dtype=bool)
    occ[np.clip((proj - proj.min()).round().astype(int), 0, span - 1)] = True
    on_runs: list[int] = []
    off_runs: list[int] = []
    run = 1
    for i in range(1, span):
        if occ[i] == occ[i - 1]:
            run += 1
            continue
        (on_runs if occ[i - 1] else off_runs).append(run)
        run = 1
    (on_runs if occ[-1] else off_runs).append(run)
    dash = None
    gaps = [g for g in off_runs if g >= 3]
    if len(on_runs) >= 3 and len(gaps) >= 2:
        on_med = float(np.median(on_runs))
        dash = "dot" if on_med <= 3.0 else "dash"
    if arrow and dash:
        return None  # dashed arrows: rare; keep the pixel-perfect PNG
    px_all = crop_bgr[fg].reshape(-1, 3).astype(np.int16)
    dist_bg = np.abs(px_all - bg[None, :]).max(axis=1)
    core = px_all[dist_bg >= np.percentile(dist_bg, 50)]
    line_bgr = np.median(core, axis=0).astype(np.int16)
    p_tail = mean + v1 * float(proj.min())
    p_head = mean + v1 * float(proj.max())
    if arrow == "start":
        p_tail, p_head = p_head, p_tail
    cx1_, cy1_ = (int(round(v)) for v in p_tail)
    cx2_, cy2_ = (int(round(v)) for v in p_head)
    cx1_ = max(0, min(w - 1, cx1_)); cx2_ = max(0, min(w - 1, cx2_))
    cy1_ = max(0, min(h - 1, cy1_)); cy2_ = max(0, min(h - 1, cy2_))
    return ((cx1_, cy1_, cx2_, cy2_), _bgr_to_hex(line_bgr),
            float(max(1.0, width_est)), dash, arrow)


def classify_elbow_line(crop_bgr: np.ndarray):
    """Lift an L-shaped (single 90° bend) connector to a polyline.

    Straight strokes route through ``classify_connector_line``; an
    elbow used to fall through to the flattened PNG path. Requires one
    dominant, axis-aligned stroke hugging two ADJACENT bbox sides with
    the opposite sides empty, and one uniform stroke colour. Returns the
    same tuple shape as ``classify_connector_line`` but with a 6-value
    ``points`` list (corner in the middle) — or None. Z-shaped (two
    bend) polylines keep the PNG path.
    """
    h, w = crop_bgr.shape[:2]
    if h < 12 or w < 12:
        return None
    border = np.concatenate([
        crop_bgr[:2, :].reshape(-1, 3),
        crop_bgr[-2:, :].reshape(-1, 3),
        crop_bgr[:, :2].reshape(-1, 3),
        crop_bgr[:, -2:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0).astype(np.int16)
    fg = np.abs(crop_bgr.astype(np.int16) - bg[None, None]).max(axis=2) > 14
    fg_u8 = cv2.morphologyEx(
        fg.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8))
    n_labels, labels = cv2.connectedComponents(fg_u8, 8)
    if n_labels - 1 == 0:
        return None
    counts = np.bincount(labels.ravel())[1:]
    total = int((fg_u8 > 0).sum())
    if total < 24 or int(counts.max()) < 0.85 * total:
        return None
    m = labels == (int(np.argmax(counts)) + 1)
    ys, xs = np.nonzero(m)
    if len(xs) < 24:
        return None
    fg_px = int(m.sum())
    stroke = fg_px / max(1.0, float(xs.max() - xs.min() + ys.max() - ys.min()))
    if stroke > 0.35 * min(h, w):
        return None  # thick blob, not a stroke
    tol = max(2, int(round(1.8 * stroke)))

    def _side_frac(top_side: bool, horizontal: bool) -> float:
        # Fraction of columns (rows) whose nearest ink to this side lies
        # within tol of it — an arm hugging the side scores ~1.0. For the
        # bottom/right sides the flipped-array argmax index IS the
        # distance from that side.
        if horizontal:
            any_ = m.any(axis=0)
            near = np.argmax(m, axis=0) if top_side \
                else np.argmax(m[::-1, :], axis=0)
            return float(((near <= tol) & any_).sum()) / max(1, w)
        any_ = m.any(axis=1)
        near = np.argmax(m, axis=1) if top_side \
            else np.argmax(m[:, ::-1], axis=1)
        return float(((near <= tol) & any_).sum()) / max(1, h)

    corner_kind = None
    for kind, (h_top, v_top) in (
        ("tl", (True, True)),
        ("tr", (True, False)),
        ("bl", (False, True)),
        ("br", (False, False)),
    ):
        if _side_frac(h_top, True) < 0.85 or _side_frac(v_top, False) < 0.85:
            continue
        # Opposite sides must be mostly empty.
        opp_h = _side_frac(not h_top, True)
        opp_v = _side_frac(not v_top, False)
        if max(opp_h, opp_v) > 0.30:
            continue
        corner_kind = kind
        break
    if corner_kind is None:
        return None
    # One uniform stroke colour, sampled from the stroke core.
    px_all = crop_bgr[m].reshape(-1, 3).astype(np.int16)
    dist_bg = np.abs(px_all - bg[None, :]).max(axis=1)
    core = px_all[dist_bg >= np.percentile(dist_bg, 50)]
    if len(core) < 12:
        return None
    line_bgr = np.median(core, axis=0).astype(np.int16)
    spread = np.abs(core - line_bgr[None, :]).max(axis=1)
    if float((spread <= 36).mean()) < 0.75:
        return None
    kind = corner_kind
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if kind == "tl":
        pts = ((x2, y1), (x1, y1), (x1, y2))
    elif kind == "tr":
        pts = ((x1, y1), (x2, y1), (x2, y2))
    elif kind == "bl":
        pts = ((x2, y2), (x1, y2), (x1, y1))
    else:
        pts = ((x1, y2), (x2, y2), (x2, y1))
    flat = []
    for px_, py_ in pts:
        flat.append(max(0, min(w - 1, px_)))
        flat.append(max(0, min(h - 1, py_)))
    return (tuple(flat), _bgr_to_hex(line_bgr),
            float(max(1.0, stroke)), None, None)


def _bg_excluding_fg(crop_bgr: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Re-sample the background colour from non-foreground border pixels.

    When a shape touches every crop edge the 2-px border ring is half
    shape ink; the plain border median can land ON the shape colour and
    invert the ring classifiers' colour-core sampling.
    """
    border_sel = np.zeros(fg.shape, dtype=bool)
    border_sel[:2, :] = border_sel[-2:, :] = True
    border_sel[:, :2] = border_sel[:, -2:] = True
    bg_px = crop_bgr[border_sel & ~fg]
    if len(bg_px) >= 12:
        return np.median(bg_px, axis=0).astype(np.int16)
    return np.array([255, 255, 255], np.int16)


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
        cand = (((gray < 250) & (diff_white > 6))
                | ((hsv[:, :, 1] > 10) & (diff_white > 4)))
        if float(cand.mean()) < 0.30:
            # Sparse silhouettes (thin rings, dashed frames) can't reach
            # the filled branches, but the ring branches may still claim
            # them — the ring classifiers carry their own structure and
            # colour gates, so a low floor is safe here. bg must be
            # re-sampled EXCLUDING foreground border pixels: the border
            # ring is half dash/shape ink here, and a shape-coloured bg
            # would invert the colour-core sampling.
            if float(cand.mean()) < 0.02:
                return None
            sparse_bg = _bg_excluding_fg(crop_bgr, cand)
            ring = _ring_candidate(crop_bgr, cand, sparse_bg)
            if ring is not None:
                return ring
            quad = _quad_ring_candidate(crop_bgr, cand, sparse_bg)
            if quad is not None:
                return quad
            return _ellipse_ring_candidate(crop_bgr, cand, sparse_bg)
        fg = cand
    elif float(fg.mean()) > 0.60:
        # Hollow frame touching the crop edge: the border ring sampled
        # the frame's stroke, inverting the mask (interior reads as
        # foreground). Re-segment against the white reference and keep
        # the result when it actually separates a sparse silhouette.
        cand = (((gray < 250) & (diff_white > 6))
                | ((hsv[:, :, 1] > 10) & (diff_white > 4)))
        if float(cand.mean()) < float(fg.mean()):
            fg = cand
            bg = _bg_excluding_fg(crop_bgr, fg)
    if float(fg.mean()) < 0.55:
        ring = _ring_candidate(crop_bgr, fg, bg)
        if ring is not None:
            return ring
        quad = _quad_ring_candidate(crop_bgr, fg, bg)
        if quad is not None:
            return quad
        ellipse = _ellipse_ring_candidate(crop_bgr, fg, bg)
        if ellipse is not None:
            return ellipse
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
    # Padding-invariant metrics: the builder hands the classifier a
    # background-padded crop (icon bbox loosening), so coverage /
    # corner / radius must be measured against the silhouette's own
    # bbox, not the padded canvas. For boundary-touching shapes the
    # tight bbox equals the crop and nothing changes.
    ys_t, xs_t = np.nonzero(filled)
    if len(xs_t) == 0:
        return None
    tx1, tx2 = int(xs_t.min()), int(xs_t.max()) + 1
    ty1, ty2 = int(ys_t.min()), int(ys_t.max()) + 1
    filled = filled[ty1:ty2, tx1:tx2]
    tw, th_ = tx2 - tx1, ty2 - ty1
    cov = float(filled.mean())
    corner = _corner_fill_fraction(filled)

    colors = _vet_fill_colors(crop_bgr[ty1:ty2, tx1:tx2],
                              fg[ty1:ty2, tx1:tx2], filled, bg)
    if colors is None:
        return None
    fill_hex, line_hex = colors
    # Convex polygons the rect/oval gates can't express (isoceles
    # triangles, diamonds, trapezoids, parallelograms, regular
    # pentagons/hexagons — cov lands between the bands).
    # Runs after the colour gates so fill/line are already vetted.
    poly = _polygon_candidate(filled)
    if poly is not None:
        return (poly, fill_hex, line_hex, 0.0, 0.0)
    # Solid block arrows (process-flow chevron arrows, step pointers)
    # are concave so the polygon gates reject them; the structural
    # stem+head profile test lifts them instead. The builder re-measures
    # shaft/head adjustments via arrow_geometry().
    arrow = _arrow_candidate(filled)
    if arrow is not None:
        return (arrow[0], fill_hex, line_hex, 0.0, 0.0)
    # Thresholds calibrated on rasterized squares with corner radius
    # r/s in [0, 0.5]: corner(c=0.12) falls monotonically 1.0 -> 0.0
    # while cov falls 1.0 -> 0.785 (circle). Mid gaps fall back to PNG.
    if cov >= 0.90 and corner >= 0.55:
        return ("rect", fill_hex, line_hex, 0.0, 0.0)
    if cov >= 0.84 and corner <= 0.45:
        # Corner radius from the area lost to rounding:
        # 1 - cov = (4 - pi) * r^2 / (w * h); the MSO adjustment is
        # r / min(w, h). Normalising by sqrt(w * h) — the old formula —
        # underestimated wide/tall rounded rects by sqrt(w / h): a 3:1
        # capsule measured 0.29 instead of a true 0.5.
        radius = math.sqrt(
            max(0.0, 1.0 - cov) * (tw * th_) / (4.0 - math.pi))
        radius = radius / max(1.0, min(th_, tw))
        radius = min(0.5, max(0.08, radius))
        return ("round_rect", fill_hex, line_hex, round(radius, 3), 0.0)
    if 0.60 <= cov <= 0.835 and corner <= 0.03:
        # The ellipse signature (cov ≈ pi/4, empty corners) is shared
        # by rotated polygons with a similar footprint; require the
        # boundary to hug one fitted ellipse before lifting.
        boundary_ring = filled & ~(
            cv2.erode(filled.astype(np.uint8) * 255,
                      np.ones((3, 3), np.uint8),
                      iterations=2) > 0)
        if _ellipse_band_fit(boundary_ring) is not None:
            return ("oval", fill_hex, line_hex, 0.0, 0.0)
    return None


def classify_outline_ring(
        source: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask_path: str | None = None,
) -> tuple[str, float] | None:
    """Classify a card-outline candidate (any aspect ratio).

    Near-square outlines are frequently circles, and a rounded rectangle
    drawn over a circular ring shows an extra visible box. This
    classifier decides the safe lift per silhouette:
      * ring pixels hug one enclosing circle -> ("oval", 0.0)
      * ring runs along all four straight sides -> ("round_rect", radius)
      * ring hugs one fitted ellipse (wide oval outline) -> ("oval", 0.0)
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
    # Stroke-aware per-column/row threshold: a min-side-scaled band is
    # much taller than a thin stroke on wide outlines (7 px band vs a
    # 3 px stroke measured 3/7 = 0.43 and failed the old half-filled
    # test). Compare the ink depth per column against the estimated
    # stroke width instead.
    stroke = float(mask.sum()) / max(1.0, 2.0 * (w + h) - 4.0)
    thr = max(1.0, 0.5 * stroke)
    def _straight(zone: np.ndarray, axis: int) -> float:
        # Fraction of columns (rows) carrying at least a stroke's depth
        # of ink. Circle tangents cross the band only near the centre
        # column; a straight card edge fills it across the whole run.
        fills = zone.sum(axis=axis)
        return float((fills >= thr).mean())
    sides = (
        _straight(mask[:band, :], 0),
        _straight(mask[-band:, :], 0),
        _straight(mask[:, :band], 1),
        _straight(mask[:, -band:], 1),
    )
    if min(sides) < 0.50:
        # Dash gaps read as missing side ink; bridge dash-scale gaps
        # along each axis (directional closes don't fatten the stroke)
        # and retry before falling through. A dashed CARD frame is a
        # common deck motif and lifts to a native dashed round-rect.
        closed9 = cv2.morphologyEx(
            mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
            np.ones((1, 11), np.uint8))
        closed9 = cv2.morphologyEx(
            closed9, cv2.MORPH_CLOSE, np.ones((11, 1), np.uint8)) > 0
        sides9 = (
            _straight(closed9[:band, :], 0),
            _straight(closed9[-band:, :], 0),
            _straight(closed9[:, :band], 1),
            _straight(closed9[:, -band:], 1),
        )
        if min(sides9) >= 0.50:
            c = max(3, int(round(min(h, w) * 0.25)))
            cys, cxs = np.nonzero(closed9[:c, :c])
            if len(cxs):
                dist = float(np.min(np.hypot(cxs, cys))) + 2.0
                radius = min(0.5, max(0.08, dist / 0.414 / min(h, w)))
            else:
                radius = 0.08
            return ("dashed_round_rect", round(radius, 3))
        # Not a straight-sided frame. A thin closed band around one
        # fitted ellipse is still a safe oval lift (wide decorative
        # ellipse outlines hug no circle, so the circle test above
        # misses them).
        if _outline_hugs_ellipse(mask):
            return ("oval", 0.0)
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
