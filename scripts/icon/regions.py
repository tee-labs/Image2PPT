"""Flat-fill color-region detection for dense flat-design slides.

Real deck renders rarely use pure-white card fills — pale tints
(#F8F6F0-class off-whites, pastel band fills) sit 10-20 channel steps
from the canvas, so the edge/fill foreground mask fuses the WHOLE
content area into one connected blob: no interior holes, no separate
border rings, and every contour/CC-based card detector dies.

On flat-design slides the reliable decomposition invariant is COLOR:
each card is one uniform fill, so connected components of each dominant
fill color reconstruct the cards — even when cards, bands, buttons and
chips all touch. ``detect_color_regions`` walks the dominant color
buckets of the text-erased image and returns one candidate per
spatially-separate same-color region:

  (x1, y1, x2, y2, color_bgr, mask_in_crop)

``mask_in_crop`` marks the region's own pixels inside its bbox; the
caller rebuilds a pristine classification crop by filling the bbox's
non-region interior holes (icons, buttons, glyph remnants) with the
region color.
"""
from __future__ import annotations

import cv2
import numpy as np


def _canvas_color(source: np.ndarray) -> np.ndarray:
    h, w = source.shape[:2]
    c = max(2, int(round(4 * (h / 720.0))))
    corners = np.concatenate([
        source[:c, :c].reshape(-1, 3),
        source[:c, -c:].reshape(-1, 3),
        source[-c:, :c].reshape(-1, 3),
        source[-c:, -c:].reshape(-1, 3),
    ]).astype(np.int16)
    return np.median(corners, axis=0).astype(np.int16)


def detect_color_regions(
    source: np.ndarray,
    *,
    ocr_text_items: list[dict] | None = None,
    scale: float = 1.0,
    min_dim: int = 20,
    min_area: int = 400,
    max_count: int = 64,
    max_buckets: int = 24,
) -> list[tuple]:
    """Find solid flat-fill regions (cards, bands, chips, chevrons).

    Returns up to ``max_count`` candidates, largest area first:
    ``(x1, y1, x2, y2, color_bgr, mask_in_crop)`` with ``mask_in_crop``
    a bool array of the FULL image size (crop-local convention used by
    the inventory builder's fill jobs).
    """
    h, w = source.shape[:2]
    canvas = _canvas_color(source)
    diff_canvas = np.abs(source.astype(np.int16) - canvas[None, None]).max(axis=2)
    fg = diff_canvas >= 10
    fg_count = int(fg.sum())
    if fg_count < 100:
        return []

    px = source[fg].reshape(-1, 3)
    quant = (px.astype(np.uint16) // 16).astype(np.uint8)
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    order = np.argsort(-counts)
    taken = 0
    regions: list[tuple] = []
    # Per-pixel quantised view for exact-bucket matching: two fills that
    # land in adjacent buckets stay DISJOINT masks by construction (a
    # median-bridge soft match would fuse a card into its near-tone
    # band). Gradient fragments are re-joined by _merge_gradient_strips.
    quant_img = (source.astype(np.uint16) // 16).astype(np.uint8)

    text_mask = np.zeros((h, w), dtype=bool)
    if ocr_text_items:
        for it in ocr_text_items:
            conf = it.get("confidence", 1.0)
            if len(it.get("text", "")) <= 2 and conf < 0.85:
                continue
            tx1 = max(0, int(it["x1"]))
            ty1 = max(0, int(it["y1"]))
            tx2 = min(w, int(it["x2"]))
            ty2 = min(h, int(it["y2"]))
            if tx2 > tx1 and ty2 > ty1:
                text_mask[ty1:ty2, tx1:tx2] = True

    for idx in order[:max_buckets * 2]:
        if taken >= max_buckets or len(regions) >= max_count:
            break
        count = int(counts[idx])
        if count < max(int(200 * scale * scale), int(0.004 * fg_count)):
            continue
        bucket = keys[idx]
        sel = (quant == bucket).all(axis=1)
        color = np.median(px[sel], axis=0).astype(np.int16)
        if float(np.max(np.abs(color - canvas))) <= 4:
            continue  # canvas-tinted AA bucket (canvas pixels are
            # already excluded by fg; don't also eat near-canvas fills)
        cmask = (quant_img == bucket[None, None]).all(axis=2) & fg
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            cmask.astype(np.uint8) * 255, 8)
        cands = []
        for i in range(1, n):
            x, y, w_, h_, area = (int(v) for v in stats[i])
            cands.append((area, x, y, w_, h_, i))
        cands.sort(reverse=True)
        for area, x, y, w_, h_, i in cands:
            if len(regions) >= max_count:
                break
            if w_ < min_dim or h_ < min_dim or area < min_area:
                continue
            density = area / float(w_ * h_)
            if density < 0.50:
                continue
            # A full-bleed region (banner touching the slide edge) is
            # legitimate; sparse border-touchers are frames/noise.
            if (x == 0 or y == 0 or x + w_ == w or y + h_ == h) \
                    and density < 0.90:
                continue
            comp = labels == i
            text_overlap = int((comp & text_mask).sum())
            if text_overlap > 0.7 * area:
                continue
            # Dedupe on MASK-pixel overlap, not bboxes: a pale card and
            # the band it sits on have close fills and heavily nested
            # bboxes, but disjoint PIXELS — both are real, both stay.
            # Two bucket fragments of one flat region share their
            # pixels — the second is a true duplicate.
            dup = False
            for ox1, oy1, ox2, oy2, _oc, ocomp in regions:
                o_area = int(ocomp.sum())
                inter = int((comp & ocomp).sum())
                if inter >= 0.50 * min(area, o_area):
                    dup = True
                    break
            if dup:
                continue
            regions.append((x, y, x + w_, y + h_, color, comp))
    regions = _merge_gradient_strips(regions)
    # Largest first: bands must be emitted (and inpainted) before the
    # cards sitting on them, so PPT z-order and inpaint order both work
    # out without extra coordination.
    regions.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    return regions


def _merge_gradient_strips(
    regions: list[tuple],
    *,
    max_rounds: int = 8,
) -> list[tuple]:
    """Re-join gradient fills split across colour buckets.

    A vertical gradient card lands in 2-3 adjacent buckets; each
    bucket's fragment is a horizontal strip of the same card. Strips
    that touch (gap ≤ 6 px), overlap across the seam, and carry
    near-identical colours (Δ ≤ 24 per channel) are unioned back into
    one region — bbox, pixel mask, and area-weighted colour.
    """
    regions = list(regions)
    for _ in range(max_rounds):
        merged_any = False
        out: list[tuple] = []
        for reg in regions:
            x1, y1, x2, y2, color, comp = reg
            hit = None
            for j, (ox1, oy1, ox2, oy2, ocolor, ocomp) in enumerate(out):
                if float(np.max(np.abs(color - ocolor))) > 24.0:
                    continue
                # Nested fills (a card on its band) also touch and can
                # be near in colour — but one bbox contains the other;
                # strips of one gradient region never do.
                ix1, iy1 = max(x1, ox1), max(y1, oy1)
                ix2, iy2 = min(x2, ox2), min(y2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    a1 = max(1, (x2 - x1) * (y2 - y1))
                    a2 = max(1, (ox2 - ox1) * (oy2 - oy1))
                    if inter_area >= 0.80 * min(a1, a2):
                        continue
                # Touching along y (horizontal strips) with sufficient
                # x overlap, or along x with sufficient y overlap.
                if y2 + 6 >= oy1 and y1 - 6 <= oy2:
                    ox_overlap = min(x2, ox2) - max(x1, ox1)
                    if ox_overlap >= 0.70 * min(x2 - x1, ox2 - ox1):
                        hit = j
                        break
                if x2 + 6 >= ox1 and x1 - 6 <= ox2:
                    oy_overlap = min(y2, oy2) - max(y1, oy1)
                    if oy_overlap >= 0.70 * min(y2 - y1, oy2 - oy1):
                        hit = j
                        break
            if hit is None:
                out.append(reg)
                continue
            mx1, my1, mx2, my2, mcolor, mcomp = out[hit]
            n_area = int(comp.sum())
            m_area = int(mcomp.sum())
            w_new = n_area / float(n_area + m_area)
            new_color = (color * w_new + mcolor * (1.0 - w_new)).astype(np.int16)
            out[hit] = (min(x1, mx1), min(y1, my1), max(x2, mx2),
                        max(y2, my2), new_color, comp | mcomp)
            merged_any = True
        regions = out
        if not merged_any:
            break
    return regions
