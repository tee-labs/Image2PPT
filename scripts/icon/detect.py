"""Sub-icon and internal-shape detection inside a parent component crop.

These detectors all operate INSIDE an already-detected parent component's
bbox and surface smaller child elements that should be split off as their
own movable image elements. Each returns the child bboxes plus a
``fill_jobs`` list describing pixels to overwrite on the parent's cleaned
image (so the parent's downstream asset crop comes out without the
extracted child baked in).

All three detectors accept ``ocr_text_items``: the pre-filtered list of
OCR items that represent actual text (i.e. callers should already have
dropped items where ``should_preserve_visual`` returns True). They use it
to suppress candidates that overlap text glyphs.
"""
from __future__ import annotations

import cv2
import numpy as np


def _sample_local_bg(
    crop: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fallback: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Sample the background immediately around a child bbox.

    Parent crops can span multiple visual zones (blue title bar plus pale
    rows, red callout plus white slide background, etc.). Using the parent
    edge colour to erase a child icon leaves obvious white/pink blocks. The
    local ring around the child is the colour we actually need for patching.
    """
    h, w = crop.shape[:2]
    pad = max(2, int(round(5 * scale)))
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(w, x2 + pad)
    ey2 = min(h, y2 + pad)
    if ex2 <= ex1 or ey2 <= ey1:
        return fallback.astype(np.uint8)

    region = crop[ey1:ey2, ex1:ex2]
    ring = np.ones(region.shape[:2], dtype=bool)
    ix1, iy1 = x1 - ex1, y1 - ey1
    ix2, iy2 = x2 - ex1, y2 - ey1
    ring[max(0, iy1):min(region.shape[0], iy2),
         max(0, ix1):min(region.shape[1], ix2)] = False
    pixels = region[ring]
    if len(pixels) < max(12, int(round(20 * scale * scale))):
        return fallback.astype(np.uint8)
    flat = pixels.reshape(-1, 3).astype(np.uint8)
    # Use the dominant quantized colour rather than the raw median. Rings
    # around icon badges often contain two colours (badge fill + host card);
    # the median can land between them, while the dominant bucket tracks the
    # actual local background we should erase back to.
    quant = (flat.astype(np.uint16) // 16).astype(np.uint8)
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    if len(keys) == 0:
        return fallback.astype(np.uint8)
    key = keys[int(np.argmax(counts))]
    cluster = flat[(quant == key).all(axis=1)]
    if len(cluster) < max(6, int(0.08 * len(flat))):
        cluster = flat
    return np.median(cluster.reshape(-1, 3), axis=0).astype(np.uint8)


def _line_visual_support(
    crop: np.ndarray,
    shape_mask: np.ndarray,
    x: int,
    y: int,
    w_: int,
    h_: int,
    fallback_bg: np.ndarray,
    scale: float,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """Grow a line-art seed to the local visible foreground island.

    The seed component usually captures coloured strokes. The actual icon
    may also include a filled badge, warning triangle, or other island that
    differs from the local background. Segment that foreground in a bounded
    local window and keep only components overlapping the seed.
    """
    ch, cw = crop.shape[:2]
    pad = max(8, int(round(20 * scale)))
    wx1 = max(0, x - pad)
    wy1 = max(0, y - pad)
    wx2 = min(cw, x + w_ + pad)
    wy2 = min(ch, y + h_ + pad)
    if wx2 <= wx1 or wy2 <= wy1:
        empty = np.zeros_like(shape_mask, dtype=bool)
        empty[y:y + h_, x:x + w_] = shape_mask[y:y + h_, x:x + w_]
        return (x, y, x + w_, y + h_), empty, fallback_bg.astype(np.uint8)

    local_bg = _sample_local_bg(crop, x, y, x + w_, y + h_,
                                fallback_bg, scale)
    win = crop[wy1:wy2, wx1:wx2]
    local_seed = shape_mask[wy1:wy2, wx1:wx2]
    gray = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV)
    diff = np.abs(win.astype(np.int16) - local_bg.astype(np.int16)).max(axis=2)
    fg = (
        (diff > 18)
        & ((hsv[:, :, 1] > 12) | (gray < 245) | (gray > 248))
    ) | local_seed
    fg = cv2.morphologyEx(
        fg.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ) > 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg.astype(np.uint8), 8)
    visual_local = np.zeros_like(fg, dtype=bool)
    seed_count = max(1, int(local_seed.sum()))
    max_area = max(80, int(4.0 * max(1, w_ * h_)))
    max_w = max(18, int(2.5 * max(1, w_)))
    max_h = max(18, int(2.5 * max(1, h_)))
    for i in range(1, n):
        cx, cy, ww, hh, area = (int(v) for v in stats[i])
        if area > max_area or ww > max_w or hh > max_h:
            continue
        component = labels == i
        overlap = int((component & local_seed).sum())
        if overlap < max(3, int(0.04 * seed_count)):
            continue
        visual_local |= component

    if int(visual_local.sum()) < max(4, int(0.25 * seed_count)):
        visual_local = local_seed.copy()

    visual = np.zeros_like(shape_mask, dtype=bool)
    visual[wy1:wy2, wx1:wx2] = visual_local
    ys, xs = np.where(visual)
    if len(xs) == 0:
        return (x, y, x + w_, y + h_), visual, local_bg
    pad_bbox = max(1, int(round(2 * scale)))
    rx1 = max(0, int(xs.min()) - pad_bbox)
    ry1 = max(0, int(ys.min()) - pad_bbox)
    rx2 = min(cw, int(xs.max()) + 1 + pad_bbox)
    ry2 = min(ch, int(ys.max()) + 1 + pad_bbox)
    return (rx1, ry1, rx2, ry2), visual, local_bg


def detect_internal_shapes(
    source: np.ndarray,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    ocr_text_items: list[dict] | None = None,
    min_dim: int = 20,
    max_dim: int = 220,
    min_area: int = 400,
    scale: float = 1.0,
) -> tuple[list[tuple], list[tuple]]:
    """Inside a parent component, find small rectangular sub-shapes that
    differ from the surrounding card background.

    Targets:
      - Numbered round badges (e.g. solid blue 1/2/3 circles on a light
        card — saturated dark fill, clearly distinct from bg).
      - Dashed-border photo placeholders (slightly darker gray than the
        host card).
      - Any solid colored UI block that lives inside a larger composition
        and would otherwise be baked into the parent's image, leaving the
        user with no way to move or replace it independently.

    Returns (bboxes, fill_jobs):
      bboxes: [(x1, y1, x2, y2), ...] in image coordinates.
      fill_jobs: [(mask_in_crop, color), ...] pixels to overwrite on the
        cleaned image with the surrounding bg colour, so the parent's
        downstream asset crop comes out without the embedded shape.
    """
    crop = source[py1:py2, px1:px2]
    h, w = crop.shape[:2]
    min_crop_dim = max(1, int(round(20 * scale)))
    if h < min_crop_dim or w < min_crop_dim:
        return [], []
    # Sample the parent's bg from a ring along the bbox edge — these
    # pixels reliably reflect the card colour because the parent is itself
    # a detected component (its bbox tightly hugs its own foreground).
    ring = max(1, int(round(3 * scale)))
    border = np.concatenate([
        crop[:ring, :].reshape(-1, 3),
        crop[-ring:, :].reshape(-1, 3),
        crop[:, :ring].reshape(-1, 3),
        crop[:, -ring:].reshape(-1, 3),
    ])
    bg_color = np.median(border, axis=0)

    # "Different from bg" — diff >= 8 per channel. Lower than the main
    # component threshold (S>12 / gray<245) so subtle inset rectangles
    # like the dashed photo placeholders (interior ~237 vs card bg ~244)
    # still surface.
    diff = np.abs(crop.astype(int) - bg_color).max(axis=2)
    different = diff >= 8

    # No morph-close: bridging gaps tends to fuse placeholder/badge with
    # adjacent row content via horizontal row separators or anti-aliased
    # fringes, producing huge low-density components that fail the size
    # filter.
    filled = different

    # OCR text mask to filter out shapes that are predominantly text.
    # EXCLUDE short low-confidence OCR results (likely badge-as-text
    # misclassifications, e.g. OCR reads a dark blue circle containing "2"
    # as "②" at conf 0.62, with the bbox covering the whole badge). If
    # we include them, every numbered badge gets blacklisted as text.
    text_mask = np.zeros((h, w), dtype=bool)
    if ocr_text_items:
        for it in ocr_text_items:
            conf = it.get("confidence", 1.0)
            if len(it["text"]) <= 2 and conf < 0.85:
                continue
            tx1 = max(0, it["x1"] - px1)
            ty1 = max(0, it["y1"] - py1)
            tx2 = min(w, it["x2"] - px1)
            ty2 = min(h, it["y2"] - py1)
            if tx2 > tx1 and ty2 > ty1:
                text_mask[ty1:ty2, tx1:tx2] = True

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled.astype(np.uint8) * 255, 8)
    shapes: list[tuple] = []
    fill_jobs: list[tuple] = []
    for i in range(1, n):
        x, y, w_, h_, area = stats[i]
        if w_ < min_dim or h_ < min_dim or w_ > max_dim or h_ > max_dim:
            continue
        if area < min_area:
            continue
        bbox_density = area / float(w_ * h_)
        if bbox_density < 0.55:
            continue
        # Discard shapes that touch the crop border — those are usually
        # the parent's own outline reaching the bbox edge, not an internal
        # element.
        if x == 0 or y == 0 or x + w_ == w or y + h_ == h:
            continue
        shape_mask = labels == i
        # Filter shapes that are almost entirely text. A single OCR text
        # bbox getting picked up here would inflate the shape count for
        # zero benefit (the text is already an editable text element).
        text_overlap = int((shape_mask & text_mask).sum())
        if text_overlap > 0.7 * area:
            continue
        shapes.append((int(px1 + x), int(py1 + y),
                       int(px1 + x + w_), int(py1 + y + h_)))
        # Fill the shape's bbox (not just the mask) so antialiased edges
        # outside the strict mask also get the bg colour. Pad ~2 px (at
        # 720-scale) to ensure clean coverage when the shape mask sits a
        # pixel inside the visible edge.
        pad = max(1, int(round(2 * scale)))
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(w, x + w_ + pad)
        ry2 = min(h, y + h_ + pad)
        rect = np.zeros((h, w), dtype=bool)
        rect[ry1:ry2, rx1:rx2] = True
        fill_jobs.append((rect, bg_color.astype(np.uint8)))
    return shapes, fill_jobs


def detect_white_subicons(
    source: np.ndarray,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    ocr_text_items: list[dict] | None = None,
    min_dim: int = 15,
    max_dim: int = 220,
    min_area: int = 200,
    scale: float = 1.0,
) -> tuple[list[tuple], list[tuple]]:
    """Inside a component bbox, find white pictograms drawn on dark uniform
    sub-shapes (e.g. a white house icon on a dark blue card that lives
    inside a larger detected component spanning multiple cards).

    Runs on the SOURCE image (not the cleaned image) so that icon pixels
    that happen to share rows with adjacent erased text are still visible.
    OCR text bboxes are passed in so that white text pixels on a dark
    card don't get mistaken for icons. A candidate whose bbox overlaps
    any OCR text bbox by >=70 % of either is dropped.

    Returns (subicons, fill_jobs):
      subicons: [(x1, y1, x2, y2), ...] in image coordinates.
      fill_jobs: [(mask_in_crop, color), ...] each describing pixels to
        overwrite on the cleaned image, with the parent dark sub-shape's
        local colour. Mask is keyed to the (py1:py2, px1:px2) crop.

    Algorithm:
      1. Inside the source crop, find DARK uniform sub-shapes (luminance
         < 100, or saturation > 80 with luminance < 180). Morph-close 5x5
         so the icon strokes don't fragment the sub-shape.
      2. Each dark sub-component IS one card. Take its local colour from
         its own pixels.
      3. The closed sub-shape mask covers the icon area too. White pixels
         inside that mask but NOT in the original dark mask are
         icon-stroke candidates — emit each connected blob as a sub-icon,
         after disqualifying any blob that's mostly inside an OCR text
         bbox.
    """
    crop = source[py1:py2, px1:px2]
    if crop.size == 0:
        return [], []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Dark uniform sub-shape pixels: very dark, or strongly saturated.
    dark_pixel = (gray < 100) | ((hsv[:, :, 1] > 80) & (gray < 180))
    dark_u8 = dark_pixel.astype(np.uint8) * 255
    # 5×5 close at 720-scale; bigger images need a bigger kernel to bridge
    # the same physical gap between dark-card fragments.
    k_dark = max(3, int(round(5 * scale)))
    if k_dark % 2 == 0:
        k_dark += 1
    kernel = np.ones((k_dark, k_dark), np.uint8)
    dark_closed = cv2.morphologyEx(dark_u8, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_closed, 8)

    # Build an "OCR text" mask in crop coordinates so we can reject icon
    # candidates that are really text glyphs.
    ch, cw = crop.shape[:2]
    text_mask = np.zeros((ch, cw), dtype=bool)
    if ocr_text_items:
        for it in ocr_text_items:
            tx1 = max(0, it["x1"] - px1)
            ty1 = max(0, it["y1"] - py1)
            tx2 = min(cw, it["x2"] - px1)
            ty2 = min(ch, it["y2"] - py1)
            if tx2 > tx1 and ty2 > ty1:
                text_mask[ty1:ty2, tx1:tx2] = True

    subicons: list[tuple] = []
    fill_jobs: list[tuple] = []
    # Looser white threshold — anti-aliased icon strokes can land at
    # luminance ~200 rather than full white. Saturation still <40 to
    # exclude faintly colored highlights.
    white_like = (gray > 195) & (hsv[:, :, 1] < 40)
    dark_min_area = max(1, int(round(1000 * scale * scale)))
    dark_min_dim = max(1, int(round(40 * scale)))
    sub_pix_min = max(1, int(round(100 * scale * scale)))
    for i in range(1, n):
        _x, _y, w_, h_, area = stats[i]
        if area < dark_min_area or w_ < dark_min_dim or h_ < dark_min_dim:
            continue
        # Require the dark sub-shape to DENSELY fill its own bounding box.
        # Only a solid filled card (e.g. a dark blue policy card, ~88 %
        # density) is a "parent card with an icon drawn inside" — in that
        # case the surrounding pixels are reliably the card's colour, so
        # the fill works. A sparse icon-shape (e.g. scales: tall thin
        # verticals with lots of light bg around them, ~30 % density) is
        # NOT a card; treating it as one fills its bbox with the icon's
        # own dark colour, painting a solid block over the actual icon
        # and the surrounding card decoration.
        bbox_density = float(area) / float(w_ * h_)
        if bbox_density < 0.7:
            continue
        sub_mask = labels == i
        original_dark_in_sub = sub_mask & dark_pixel
        sub_pixels = crop[original_dark_in_sub]
        if len(sub_pixels) < sub_pix_min:
            continue
        parent_color = np.median(sub_pixels, axis=0).astype(np.uint8)
        # Subtract OCR text from the white candidate BEFORE the
        # morph-close. Otherwise the close bridges nearby text glyphs
        # into the actual icon, producing one huge text-dominated blob
        # that fails the text-overlap rejection.
        candidate = sub_mask & white_like & ~dark_pixel & ~text_mask
        if not candidate.any():
            continue
        # White icon strokes are thin and broken up at antialiased edges.
        # A large close kernel merges roof + columns + base of typical
        # iconography into one blob even with 20+ px gaps between
        # elements (at 720-scale; bigger images use a proportionally
        # bigger kernel to bridge the same physical span).
        k_close = max(3, int(round(25 * scale)))
        if k_close % 2 == 0:
            k_close += 1
        cand_closed = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            np.ones((k_close, k_close), np.uint8))
        wn, wlabels, wstats, _ = cv2.connectedComponentsWithStats(
            cand_closed, 8)
        # Per dark sub-shape, accept the LARGEST qualifying blob as "the"
        # icon (most cards carry one icon). Multiple comparable blobs
        # would be unusual; the loop just picks the biggest above
        # min_area.
        sub_w = int(stats[i, 2])
        sub_h = int(stats[i, 3])
        best = None
        for j in range(1, wn):
            wx, wy, ww, wh, wa = wstats[j]
            if wa < min_area:
                continue
            if not (min_dim <= ww <= max_dim and min_dim <= wh <= max_dim):
                continue
            # Reject "subicons" that span most of the parent dark
            # sub-shape — those aren't an icon embedded in a card,
            # they're the dark sub-shape itself being misclassified
            # (for example, a decorative glyph plus adjacent label text can
            # close into one banner-wide blob, and that blob would replace
            # the whole banner). A genuine pictogram-inside-card is much
            # smaller than its host (typically <50 % of card width).
            if ww > 0.7 * sub_w or wh > 0.7 * sub_h:
                continue
            if best is None or wa > best[4]:
                best = (wx, wy, ww, wh, wa)
        if best is None:
            continue
        wx, wy, ww, wh, _ = best
        # Pad the bbox slightly (~3 px at 720-scale) so antialiased edge
        # pixels just outside the morph result also get covered by the
        # rect fill.
        pad = max(1, int(round(3 * scale)))
        rx1 = max(0, wx - pad)
        ry1 = max(0, wy - pad)
        rx2 = min(int(cand_closed.shape[1]), wx + ww + pad)
        ry2 = min(int(cand_closed.shape[0]), wy + wh + pad)
        subicons.append((int(px1 + rx1), int(py1 + ry1),
                         int(px1 + rx2), int(py1 + ry2)))
        # Fill the icon's padded bbox with the parent's solid colour so
        # the parent card's downstream asset crop comes out icon-free.
        rect_mask = np.zeros((ch, cw), dtype=bool)
        rect_mask[ry1:ry2, rx1:rx2] = True
        fill_jobs.append((rect_mask, parent_color))
    return subicons, fill_jobs


def detect_line_art_subicons(
    image: np.ndarray,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    ocr_text_items: list[dict] | None = None,
    min_dim: int = 12,
    max_dim: int = 120,
    min_area: int = 35,
    scale: float = 1.0,
) -> tuple[list[tuple], list[tuple]]:
    """Find sparse coloured line-art children inside a parent visual.

    Dense-shape detection intentionally ignores low-density strokes. This
    pass covers shield/cloud/lock-style pictograms drawn inside pale
    circle containers or light cards. It runs inside an already-detected
    parent crop, so the parent's own border can be rejected by dropping
    components that touch the crop edge.
    """
    crop = image[py1:py2, px1:px2]
    if crop.size == 0:
        return [], []
    ch, cw = crop.shape[:2]
    min_crop_dim = max(1, int(round(24 * scale)))
    if ch < min_crop_dim or cw < min_crop_dim:
        return [], []

    ring_lo = max(1, int(round(2 * scale)))
    ring_hi = max(ring_lo, int(round(5 * scale)))
    ring = max(ring_lo, min(ring_hi, min(ch, cw) // 12))
    border = np.concatenate([
        crop[:ring, :].reshape(-1, 3),
        crop[-ring:, :].reshape(-1, 3),
        crop[:, :ring].reshape(-1, 3),
        crop[:, -ring:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0) if len(border) else np.array([255, 255, 255])

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_bg = np.abs(crop.astype(int) - bg.astype(int)).max(axis=2)

    text_mask = np.zeros((ch, cw), dtype=bool)
    if ocr_text_items:
        for it in ocr_text_items:
            tx1 = max(0, it["x1"] - px1)
            ty1 = max(0, it["y1"] - py1)
            tx2 = min(cw, it["x2"] - px1)
            ty2 = min(ch, it["y2"] - py1)
            if tx2 > tx1 and ty2 > ty1:
                text_mask[ty1:ty2, tx1:tx2] = True

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    blueish = (hue >= 85) & (hue <= 130) & (sat > 24) & (diff_bg > 12)
    coloured_line = (diff_bg > 28) & ((sat > 24) | (gray < 220))
    edges = cv2.Canny(gray, 40, 120) > 0
    edge_line = edges & (diff_bg > 12) & ((sat > 16) | (gray < 238))
    candidate = (blueish | coloured_line | edge_line) & ~text_mask

    cand_u8 = candidate.astype(np.uint8) * 255
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE,
                               np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, 8)

    subicons: list[tuple] = []
    fill_jobs: list[tuple] = []
    kept: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w_, h_, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        if w_ < min_dim or h_ < min_dim:
            continue
        if w_ > max_dim or h_ > max_dim:
            continue
        if x <= 1 or y <= 1 or x + w_ >= cw - 1 or y + h_ >= ch - 1:
            continue
        aspect = w_ / max(1, h_)
        if aspect > 4.5 or aspect < 0.22:
            continue
        bbox_area = max(1, w_ * h_)
        density = area / float(bbox_area)
        if density > 0.72:
            continue
        shape_mask = labels == i
        if int((shape_mask & text_mask).sum()) > 0.35 * area:
            continue

        duplicate = False
        for ex1, ey1, ex2, ey2 in kept:
            ix1, iy1 = max(x, ex1), max(y, ey1)
            ix2, iy2 = min(x + w_, ex2), min(y + h_, ey2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.55 * min(bbox_area, (ex2 - ex1) * (ey2 - ey1)):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((x, y, x + w_, y + h_))

        (rx1, ry1, rx2, ry2), visual_mask, local_bg = _line_visual_support(
            crop, shape_mask, x, y, w_, h_, bg.astype(np.uint8), scale)
        subicons.append((int(px1 + rx1), int(py1 + ry1),
                         int(px1 + rx2), int(py1 + ry2)))

        # Patch the actual visible foreground support with the LOCAL
        # background. This removes filled badges and line strokes from the
        # parent without painting a rectangular block over row/card details.
        fill_mask = cv2.dilate(
            visual_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=max(1, int(round(1 * scale))),
        ).astype(bool)
        fill_jobs.append((fill_mask, local_bg))
    return subicons, fill_jobs
