#!/usr/bin/env python
"""Build a complete element inventory from a cleaned slide image + OCR JSON.

Inventory format (JSON list):
  [
    {"id":"t000","type":"text","text":"...","bbox":[x1,y1,x2,y2],"confidence":1.0},
    {"id":"v000","type":"image","bbox":[x1,y1,x2,y2],"source":"cleaned"},
    ...
  ]

The inventory captures EVERY visible element on the slide. Text comes from
OCR; images come from cv2 connected-component detection on the text-erased
image. The inventory is the single source of truth for downstream PPTX build.

Detection rules:
- Min icon area = 300 px, min dimension = 18 px (allows independent small
  icons but filters noise).
- Morphological close 6x6 BEFORE detection - merges fragments WITHIN one icon
  (e.g. small parts of a complex icon) into a single component, while leaving
  spatially separate independent icons as distinct components.
- Conservative split: each component is checked for white-space gaps >= 10 px
  internally. If found, split along those gaps. If no clear gap, KEEP WHOLE
  (preserves bar charts with arrows, complex compositions).

Usage:
    python build_inventory.py --clean clean.png --ocr ocr.json --out inv.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Re-use erase_text logic for the icon/logo filters, and the icon
# package for sub-shape detection inside parent components.
import sys
_SCRIPTS = Path(__file__).resolve().parent          # .../scripts/pipeline
_SCRIPTS_ROOT = _SCRIPTS.parent                      # .../scripts
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS_ROOT))
from erase_text import (
    detect_logo_strips,
    is_in_logo_zone,
    is_likely_icon,
    preprocess_ocr,
    should_preserve_visual,
)
from _heuristics import pixel_scale, s_area, s_kernel, s_length
from icon import (
    detect_internal_shapes,
    detect_line_art_subicons,
    detect_white_subicons,
    inpaint_region_inplace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inventory from cleaned image + OCR.")
    parser.add_argument("--clean", required=True, help="Cleaned image path (text erased).")
    parser.add_argument("--source", required=True, help="Original source image (for icon filters).")
    parser.add_argument("--ocr", required=True, help="OCR JSON path.")
    parser.add_argument("--out", required=True, help="Inventory JSON output path.")
    parser.add_argument("--min-area", type=int, default=80,
                        help="Min component area (very loose - only filters noise dots).")
    parser.add_argument("--dilate", type=int, default=6)
    parser.add_argument("--split-gap", type=int, default=12)
    parser.add_argument("--debug-dir",
                        help="Optional debug directory. Writes <stem>_inv.png "
                             "showing detected components, subicons, and OCR "
                             "text overlaid on the source image.")
    parser.add_argument("--use-fastsam", action="store_true",
                        help="Use FastSAM instance segmentation for image "
                             "component detection instead of cv2 connected "
                             "components. Each component carries a precise "
                             "mask that downstream cropping uses as an alpha "
                             "channel, so asset PNGs come out with a real "
                             "transparent background rather than a rectangle "
                             "with the host slide's bg colour. Requires "
                             "ultralytics + FastSAM-s.pt model (~23 MB).")
    parser.add_argument("--masks-dir",
                        help="When --use-fastsam is set, save per-component "
                             "mask PNGs to this directory. inventory_to_layout "
                             "reads them by element id (e.g. v005.mask.png) "
                             "to alpha-key the asset crop.")
    return parser.parse_args()


def _estimate_background_bgr(img: np.ndarray) -> np.ndarray:
    """Estimate canvas background from the four corners in BGR space."""
    h, w = img.shape[:2]
    patch = max(8, min(h, w) // 16)
    samples = [
        img[:patch, :patch],
        img[:patch, w - patch:w],
        img[h - patch:h, :patch],
        img[h - patch:h, w - patch:w],
    ]
    pixels = np.concatenate([s.reshape(-1, 3) for s in samples if s.size])
    if pixels.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    quant = (pixels // 16) * 16
    winner, _ = Counter(map(tuple, quant)).most_common(1)[0]
    winner_arr = np.array(winner, dtype=np.int16)
    diff = np.abs(pixels.astype(np.int16) - winner_arr).max(axis=1)
    close = pixels[diff <= 24]
    return np.median(close if len(close) else pixels, axis=0).astype(np.uint8)


def _foreground_mask(cleaned: np.ndarray, dilate_size: int) -> np.ndarray:
    """Foreground mask used by detect_components and conservative_split.

    Threshold: noticeably different from the corner-estimated canvas
    background plus a Canny edge layer that catches 1-px pale outlines.
    The old white-background test made every pixel of a dark slide look
    like foreground, merging the whole page into one flattened object.
    """
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    bg = _estimate_background_bgr(cleaned).astype(np.int16)
    diff_bg = np.abs(cleaned.astype(np.int16) - bg).max(axis=2)
    bg_pixel = bg.astype(np.uint8).reshape(1, 1, 3)
    bg_hsv = cv2.cvtColor(bg_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    sat_delta = np.abs(hsv[:, :, 1].astype(np.int16) - int(bg_hsv[1]))
    mask = ((diff_bg >= 8) | ((diff_bg >= 5) & (sat_delta > 18))).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, edges)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_components(cleaned: np.ndarray, min_area: int, dilate_size: int) -> list[tuple]:
    """Detect ALL connected components. Returns components above noise floor.

    Even a whole-image-sized component is returned (not filtered) so that
    conservative_split can break it apart along white-space gaps. Filtering
    "big" components prematurely loses real content (e.g. when a faint
    decorative element bridges otherwise distinct icons).

    Foreground threshold: see _foreground_mask. Catches:
    - Light fill rects (light blue cards, gray box backgrounds)
    - Subtle line art (white-bg sketches with light strokes)
    - Saturated colored regions (bars, banners, icons)
    """
    mask = _foreground_mask(cleaned, dilate_size)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        out.append((int(x), int(y), int(x + ww), int(y + hh), int(area)))
    return out


def _find_gaps(line_has_fg: np.ndarray, min_gap: int) -> list[tuple]:
    """Return list of (start, end) gap intervals in a 1D foreground signal."""
    gaps = []
    in_gap = False
    gs = 0
    n = len(line_has_fg)
    for i in range(n):
        if not line_has_fg[i]:
            if not in_gap:
                gs = i
                in_gap = True
        else:
            if in_gap:
                if i - gs >= min_gap:
                    gaps.append((gs, i))
                in_gap = False
    return gaps


def _split_axis(fg: np.ndarray, axis: int, min_gap: int) -> list[tuple]:
    """Split foreground mask along an axis using white-space gaps. Axis 0
    means horizontal slices (gaps in rows → top/bottom split); axis 1 means
    vertical slices (gaps in cols → left/right split). Returns list of
    (start, end) ranges along the OPPOSITE axis covering each foreground
    block."""
    H, W = fg.shape
    has_fg = fg.sum(axis=axis) > 0
    gaps = _find_gaps(has_fg, min_gap)
    if not gaps:
        return []
    pieces = []
    prev = 0
    for g_s, g_e in gaps:
        if g_s - prev > 0:
            pieces.append((prev, g_s))
        prev = g_e
    last = W if axis == 0 else H
    if last - prev > 0:
        pieces.append((prev, last))
    return pieces


def detect_outline_mask(cleaned: np.ndarray,
                        x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """Return an alpha mask for a large pale rectangular card outline.

    Splitting parents along whitespace can throw away the outer 1-2 px
    rounded border of a big card because that border is sparse and split
    away from the interior content. This helper keeps only pixels near
    the raw component perimeter, so the resulting asset restores the
    border without duplicating every child icon and panel inside.

    Returns None when the candidate isn't a plausible card outline:
    too small, no edge coverage, or coverage missing on too many sides.
    Big rectangles (≥250×120 at 720-scale) get more permissive
    thresholds since a rectangle this large with even sparse edge
    coverage on 3+ sides is almost certainly a real card border.
    """
    crop = cleaned[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = pixel_scale(cleaned)
    if w < s_length(180, scale) or h < s_length(80, scale):
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(crop.astype(int) - 255).max(axis=2)
    fg = (
        ((gray < 253) & (diff_white > 5))
        | ((hsv[:, :, 1] > 8) & (diff_white > 4))
    )
    band_lo = s_length(4, scale)
    band_hi = s_length(8, scale)
    band = max(band_lo, min(band_hi, int(round(min(w, h) * 0.03))))
    edge_zone = np.zeros((h, w), dtype=bool)
    edge_zone[:band, :] = True
    edge_zone[-band:, :] = True
    edge_zone[:, :band] = True
    edge_zone[:, -band:] = True
    edge = fg & edge_zone
    # `80` is a hard pixel floor on edge coverage; the `0.003 × w × h`
    # term is a fractional floor that already auto-scales with the
    # crop area. Only the hard floor needs scaling by area.
    if int(edge.sum()) < max(s_area(80, scale), int(0.003 * w * h)):
        return None

    top = float(np.any(fg[:band, :], axis=0).sum()) / max(1, w)
    bottom = float(np.any(fg[-band:, :], axis=0).sum()) / max(1, w)
    left = float(np.any(fg[:, :band], axis=1).sum()) / max(1, h)
    right = float(np.any(fg[:, -band:], axis=1).sum()) / max(1, h)
    big = (w >= s_length(250, scale)) and (h >= s_length(120, scale))
    side_floor = 0.22 if big else 0.28
    max_v_floor = 0.30 if big else 0.42
    max_h_floor = 0.22 if big else 0.32
    side_count = sum(v >= side_floor for v in (top, bottom, left, right))
    if side_count < 3 or max(top, bottom) < max_v_floor or max(left, right) < max_h_floor:
        return None

    alpha = (edge.astype(np.uint8) * 255)
    alpha = cv2.dilate(alpha, np.ones((2, 2), np.uint8), iterations=1)
    return alpha


def detect_outline_rects(cleaned: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Find nested large card outlines that a single huge component
    would otherwise hide. detect_components returns one merged CC for a
    multi-card panel; this function uses contour analysis on the same
    foreground mask to surface each card's border as its own outline
    record with a ring-shaped alpha mask.
    """
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_BGR2HSV)
    diff_white = np.abs(cleaned.astype(int) - 255).max(axis=2)
    mask = (
        ((gray < 253) & (diff_white > 5))
        | ((hsv[:, :, 1] > 8) & (diff_white > 4))
    ).astype(np.uint8) * 255
    # Edge layer also feeds the contour pass so faint borders that
    # don't pass the fill threshold still get surfaced as outlines.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, edges)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    img_h, img_w = cleaned.shape[:2]
    scale = pixel_scale(cleaned)
    min_w = s_length(180, scale)
    min_h = s_length(80, scale)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_w or h < min_h:
            continue
        if w > img_w * 0.96 or h > img_h * 0.90:
            continue
        alpha = detect_outline_mask(cleaned, x, y, x + w, y + h)
        if alpha is None:
            continue
        candidates.append((int(x), int(y), int(x + w), int(y + h), alpha))

    # Prefer the tighter nested contour over an external contour that also
    # includes a tab label or adjacent connector.
    out: list[tuple[int, int, int, int, np.ndarray]] = []
    for cand in sorted(candidates,
                       key=lambda r: (r[2] - r[0]) * (r[3] - r[1])):
        x1, y1, x2, y2, _ = cand
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for ox1, oy1, ox2, oy2, _om in out:
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            similar_size = min(area, oarea) / max(area, oarea) >= 0.55
            if similar_size and (ix2 - ix1) * (iy2 - iy1) >= 0.85 * min(area, oarea):
                duplicate = True
                break
        if not duplicate:
            out.append(cand)
    return out


def conservative_split(crop: np.ndarray, min_gap: int = 12) -> list[tuple]:
    """Split via white-space gaps in BOTH horizontal and vertical directions.
    Tries vertical-strip split first (left-right), then for each strip tries
    horizontal split (top-bottom). Any clean gap of `min_gap` px or more is
    accepted — there is no sub-piece size threshold. The previous
    `min_split_dim` guard removed: it suppressed legitimate splits when an
    icon was visually small but cleanly separated from its neighbour.

    Returns [(x1,y1,x2,y2), ...] relative to crop. Single-element list when
    no clean split is possible."""
    H, W = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    fg = ((gray < 225) | (hsv[:, :, 1] > 30)).astype(np.uint8)

    def fit_bbox(strip):
        rs = np.where(strip.sum(axis=1) > 0)[0]
        cs = np.where(strip.sum(axis=0) > 0)[0]
        if len(rs) and len(cs):
            return (int(cs[0]), int(rs[0]), int(cs[-1]) + 1, int(rs[-1]) + 1)
        return None

    # Vertical-strip split (left-right) first
    col_pieces = _split_axis(fg, axis=0, min_gap=min_gap)
    if not col_pieces:
        col_pieces = [(0, W)]

    sub_boxes = []
    for cs, ce in col_pieces:
        col_strip = fg[:, cs:ce]
        # Within each column-strip, try horizontal split (top-bottom)
        row_pieces = _split_axis(col_strip, axis=1, min_gap=min_gap)
        if not row_pieces:
            row_pieces = [(0, H)]
        for rs, re in row_pieces:
            block = col_strip[rs:re]
            tight = fit_bbox(block)
            if tight is None:
                continue
            sub_boxes.append((cs + tight[0], rs + tight[1],
                              cs + tight[2], rs + tight[3]))

    if len(sub_boxes) <= 1:
        return [(0, 0, W, H)]
    return sub_boxes


def main() -> None:
    args = parse_args()
    cleaned = cv2.imread(args.clean)
    source = cv2.imread(args.source)
    if cleaned is None or source is None:
        raise SystemExit("Could not load images.")
    ocr_data = json.loads(Path(args.ocr).read_text(encoding="utf-8"))
    ocr_data = preprocess_ocr(ocr_data, img=source)
    # Pre-filter OCR for the icon detectors: they want only real text
    # glyphs (so a glyph bbox can mask off candidate pixels), not
    # OCR-misclassified visuals like numbered badges read as "②".
    ocr_text_items = [it for it in ocr_data if not should_preserve_visual(it)]
    h = cleaned.shape[0]
    # Every pixel constant in this pipeline was tuned at 720-tall source.
    # `scale` rescales them so 480p / 1080p / 4K inputs all filter
    # equivalent physical content. `args.min_area` (area), `args.dilate`
    # (kernel), and `args.split_gap` (length) are user-facing but in
    # practice always run at their defaults — silently scaling them keeps
    # default-flow runs correct without surprising explicit users.
    scale = pixel_scale(cleaned)
    args.min_area = s_area(args.min_area, scale)
    args.dilate = s_kernel(args.dilate, scale)
    args.split_gap = s_length(args.split_gap, scale)

    # Save a sidecar TEXT-ONLY-erased cleaned image BEFORE we modify
    # `cleaned` in place with shape-area fills. Downstream asset cropping
    # of subicons / internal shapes uses this sidecar — it has the OCR
    # text already erased but the shapes themselves intact, so the cropped
    # asset doesn't carry duplicate white text that would render on top
    # of the editable text element placed at the same position.
    clean_path = Path(args.clean)
    text_only_path = clean_path.with_name(f"{clean_path.stem}.text_only.png")
    cv2.imwrite(str(text_only_path), cleaned)

    # Pre-filter OCR for likely-icons and logo-band texts (these never become editable text).
    logo_bands = detect_logo_strips(ocr_data, h)
    candidate_texts = [
        item for item in ocr_data
        if not is_likely_icon(item, ocr_data, source)[0]
        and not is_in_logo_zone(item, logo_bands)
    ]

    # Detect visual components — either via cv2 connected components on the
    # cleaned image (default) or via FastSAM instance segmentation on the
    # source image (when --use-fastsam is set). FastSAM is slower (~1-2s
    # per page on CPU vs ~50ms for cv2) but gives per-element masks, which
    # downstream cropping uses as an alpha channel — asset PNGs come out
    # with a true transparent background instead of a rectangle filled
    # with the host slide's bg colour.
    fastsam_segments: list[dict] = []
    if args.use_fastsam:
        from fastsam_detect import segment as _fastsam_segment
        fastsam_segments = _fastsam_segment(args.source, ocr_data=ocr_data)
        # Synthesise a `components`-shape list (x1, y1, x2, y2, area)
        # so the rest of the loop is unchanged.
        components = [
            (s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3],
             int(s["mask"].sum() // 255) if s["mask"].dtype == np.uint8
             else int(s["mask"].sum()))
            for s in fastsam_segments
        ]
    else:
        components = detect_components(cleaned, args.min_area, args.dilate)

    inventory = []

    # Text elements
    for i, item in enumerate(candidate_texts):
        entry = {
            "id": f"t{i:03d}",
            "type": "text",
            "text": item["text"],
            "bbox": [item["x1"], item["y1"], item["x2"], item["y2"]],
            "confidence": item.get("confidence", 1.0),
        }
        # Pass through PaddleOCR's per-segment data (return_word_box=True).
        # Two parallel exposures: chars/char_boxes (per-char, populated by
        # ocr_paddle only when each PP-OCRv5 word IS one character — i.e.
        # pure-CJK lines) and words/word_boxes (per-segment, always present
        # — `524个` arrives as ['524','个']). Downstream uses words first
        # for color sampling, falling back to chars for finer per-glyph
        # control on pure-CJK colour-emphasis lines.
        if "chars" in item and "char_boxes" in item:
            entry["chars"] = list(item["chars"])
            entry["char_boxes"] = [list(b) for b in item["char_boxes"]]
        if "words" in item and "word_boxes" in item:
            entry["words"] = list(item["words"])
            entry["word_boxes"] = [list(b) for b in item["word_boxes"]]
        inventory.append(entry)

    # Image components are emitted independently of nearby text. Auto-merging
    # text + adjacent icons into "logo" composites was tried and removed:
    # it confused decorative section markers for brand logos. Logo grouping
    # is now left to manual edit in PowerPoint after build.
    visual_idx = 0
    img_h, img_w = cleaned.shape[:2]
    background_records: list[tuple] = []  # whole-image fallbacks
    outline_records: list[tuple] = []     # large card borders + ring alpha
    foreground_records: list[tuple] = []  # everything else
    subicon_records: list[tuple] = []     # white-on-dark sub-icons
    line_subicon_records: list[tuple] = []  # sparse line-art child icons
    internal_shape_records: list[tuple] = []  # numbered badges, placeholders

    # Pre-compute scaled gates used by the three `_scan_*_inplace`
    # helpers and the icon detectors. Kwargs (min_dim/max_dim/min_area)
    # pass through unchanged — the detectors apply `scale` to their own
    # internal constants and to the kwarg defaults via the same factor.
    parent_min_side = s_length(120, scale)
    line_min_area = s_area(1200, scale)
    line_min_side = s_length(24, scale)

    def _box4(record: tuple) -> tuple[int, int, int, int]:
        return (int(record[0]), int(record[1]),
                int(record[2]), int(record[3]))

    def _scan_internal_shapes_inplace(sx1: int, sy1: int,
                                      sx2: int, sy2: int) -> None:
        """Detect internal sub-shapes (badges, placeholders, small colored
        UI blocks) inside this parent component, emit each as its own
        movable image element, and fill its bbox in the cleaned image with
        the parent's edge bg colour so the parent's downstream asset crop
        comes out clean. Only runs on parents big enough to plausibly
        contain such sub-shapes (≥120×120 at 720-scale) — small
        components are unlikely to host distinct internal UI elements."""
        if sx2 - sx1 < parent_min_side or sy2 - sy1 < parent_min_side:
            return
        shapes, fill_jobs = detect_internal_shapes(
            source, sx1, sy1, sx2, sy2,
            ocr_text_items=ocr_text_items,
            min_dim=s_length(20, scale),
            max_dim=s_length(220, scale),
            min_area=s_area(400, scale),
            scale=scale,
        )
        if not shapes:
            return
        local = cleaned[sy1:sy2, sx1:sx2]
        for (ix1, iy1, ix2, iy2), (mask_in_crop, color) in zip(shapes, fill_jobs):
            # Same dedup rule as subicons: skip if already covered.
            duplicate = False
            for ex1, ey1, ex2, ey2 in (
                [_box4(r) for r in subicon_records]
                + [_box4(r) for r in line_subicon_records]
                + [_box4(r) for r in internal_shape_records]
            ):
                ox1, oy1 = max(ix1, ex1), max(iy1, ey1)
                ox2, oy2 = min(ix2, ex2), min(iy2, ey2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                inter = (ox2 - ox1) * (oy2 - oy1)
                a1 = (ix2 - ix1) * (iy2 - iy1)
                a2 = (ex2 - ex1) * (ey2 - ey1)
                if inter >= 0.5 * min(a1, a2):
                    duplicate = True
                    break
            if duplicate:
                continue
            internal_shape_records.append((ix1, iy1, ix2, iy2))
            # Background-aware patch: flat-fill locally uniform cards to
            # avoid icon ghosts, but fall back to inpaint on non-uniform
            # card fills. Re-take the view each iteration in case a prior
            # patch reallocated the array.
            local = cleaned[sy1:sy2, sx1:sx2]
            inpaint_region_inplace(local, mask_in_crop, scale=scale,
                                   fill_color=color)

    def _scan_subicons_inplace(sx1: int, sy1: int, sx2: int, sy2: int) -> None:
        """Detect white pictograms inside any dark uniform sub-shape that
        lives within the given bbox, and fill each icon's pixels with that
        sub-shape's local colour so the parent card's downstream asset crop
        is icon-free. Detection runs on the SOURCE image so icon pixels
        that overlap erased text rows remain visible.

        Components can overlap (esp. dense slides with one big component
        plus several smaller ones covering the same dark card), so a
        subicon found here is dropped if it overlaps any already-emitted
        subicon by >=50 % IoU — otherwise the same icon gets extracted
        twice and rendered on top of itself."""
        subs, fill_jobs = detect_white_subicons(
            source, sx1, sy1, sx2, sy2,
            ocr_text_items=ocr_text_items,
            min_dim=s_length(15, scale),
            max_dim=s_length(220, scale),
            min_area=s_area(200, scale),
            scale=scale,
        )
        if not subs:
            return
        local = cleaned[sy1:sy2, sx1:sx2]
        for (ix1, iy1, ix2, iy2), (mask_in_crop, color) in zip(subs, fill_jobs):
            duplicate = False
            for (ex1, ey1, ex2, ey2) in subicon_records:
                ox1, oy1 = max(ix1, ex1), max(iy1, ey1)
                ox2, oy2 = min(ix2, ex2), min(iy2, ey2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                inter = (ox2 - ox1) * (oy2 - oy1)
                a1 = (ix2 - ix1) * (iy2 - iy1)
                a2 = (ex2 - ex1) * (ey2 - ey1)
                if inter >= 0.5 * min(a1, a2):
                    duplicate = True
                    break
            if duplicate:
                continue
            subicon_records.append((ix1, iy1, ix2, iy2))
            local = cleaned[sy1:sy2, sx1:sx2]
            inpaint_region_inplace(local, mask_in_crop, scale=scale,
                                   fill_color=color)

    def _scan_line_subicons_inplace(sx1: int, sy1: int,
                                    sx2: int, sy2: int) -> None:
        """Detect sparse line-art children inside this parent bbox."""
        sw = sx2 - sx1
        sh = sy2 - sy1
        if sw * sh < line_min_area or min(sw, sh) < line_min_side:
            return
        subs, fill_jobs = detect_line_art_subicons(
            source, sx1, sy1, sx2, sy2,
            ocr_text_items=ocr_text_items,
            min_dim=s_length(12, scale),
            max_dim=s_length(120, scale),
            min_area=s_area(35, scale),
            scale=scale,
        )
        if not subs:
            return
        for (ix1, iy1, ix2, iy2), (mask_in_crop, _color) in zip(subs, fill_jobs):
            duplicate = False
            for ex1, ey1, ex2, ey2 in (
                [_box4(r) for r in subicon_records]
                + [_box4(r) for r in line_subicon_records]
                + [_box4(r) for r in internal_shape_records]
            ):
                ox1, oy1 = max(ix1, ex1), max(iy1, ey1)
                ox2, oy2 = min(ix2, ex2), min(iy2, ey2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                inter = (ox2 - ox1) * (oy2 - oy1)
                a1 = (ix2 - ix1) * (iy2 - iy1)
                a2 = (ex2 - ex1) * (ey2 - ey1)
                if inter >= 0.5 * min(a1, a2):
                    duplicate = True
                    break
            if duplicate:
                continue
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[sy1:sy2, sx1:sx2] = (
                mask_in_crop.astype(np.uint8) * 255
            )
            line_subicon_records.append((ix1, iy1, ix2, iy2, full_mask))
            local = cleaned[sy1:sy2, sx1:sx2]
            inpaint_region_inplace(local, mask_in_crop, scale=scale,
                                   fill_color=_color)

    def _append_outline_record(x1: int, y1: int, x2: int, y2: int,
                               outline_mask: np.ndarray) -> None:
        """Dedup against existing outlines before appending. Two contour
        passes (per-CC detect_outline_mask + global detect_outline_rects)
        can surface the same card border, so an IoU-ish check keeps the
        inventory tidy."""
        area = max(1, (x2 - x1) * (y2 - y1))
        for ox1, oy1, ox2, oy2, _om in outline_records:
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            similar_size = min(area, oarea) / max(area, oarea) >= 0.55
            if similar_size and (ix2 - ix1) * (iy2 - iy1) >= 0.85 * min(area, oarea):
                return
        outline_records.append((x1, y1, x2, y2, outline_mask))

    # FastSAM segments are already precise per-element; conservative split
    # and the subicon/internal-shape helpers all exist to compensate for
    # the cv2 detector's coarseness, so skip them when SAM is in charge.
    foreground_to_mask: dict[tuple, np.ndarray] = {}
    if args.use_fastsam:
        for seg in fastsam_segments:
            x1, y1, x2, y2 = seg["bbox"]
            foreground_records.append((x1, y1, x2, y2))
            foreground_to_mask[(x1, y1, x2, y2)] = seg["mask"]
    else:
        # Global contour pass first — surfaces card outlines that a single
        # merged CC would otherwise hide (page-10's three pale-blue
        # workflow cards register as one CC but are three distinct cards).
        for x1, y1, x2, y2, outline_mask in detect_outline_rects(cleaned):
            _append_outline_record(x1, y1, x2, y2, outline_mask)
        for x1, y1, x2, y2, _area in components:
            crop = cleaned[y1:y2, x1:x2]
            # Per-CC outline check: when the component IS a card outline
            # (pale border + sparse interior), keep its ring as an
            # alpha-masked element even after conservative_split fragments
            # the interior.
            outline_mask = detect_outline_mask(cleaned, x1, y1, x2, y2)
            if outline_mask is not None:
                _append_outline_record(x1, y1, x2, y2, outline_mask)
            sub = conservative_split(crop, min_gap=args.split_gap)
            if len(sub) == 1:
                cw, ch = x2 - x1, y2 - y1
                if cw >= img_w * 0.95 and ch >= img_h * 0.95:
                    # Whole-image component that couldn't be split: keep it
                    # as a BACKGROUND. Smaller foreground components (already
                    # detected separately above) render on top via z-order.
                    background_records.append((x1, y1, x2, y2))
                    continue
                foreground_records.append((x1, y1, x2, y2))
                _scan_subicons_inplace(x1, y1, x2, y2)
                _scan_line_subicons_inplace(x1, y1, x2, y2)
                _scan_internal_shapes_inplace(x1, y1, x2, y2)
            else:
                for sx1, sy1, sx2, sy2 in sub:
                    ax1, ay1, ax2, ay2 = x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2
                    foreground_records.append((ax1, ay1, ax2, ay2))
                    _scan_subicons_inplace(ax1, ay1, ax2, ay2)
                    _scan_line_subicons_inplace(ax1, ay1, ax2, ay2)
                    _scan_internal_shapes_inplace(ax1, ay1, ax2, ay2)

    # Drop foreground records whose bbox is largely covered by a sub-icon or
    # internal shape. Both detectors emit a separate movable element AND
    # overwrite the cleaned image at that area with a flat bg colour, so the
    # original foreground component (cropped from the post-fill cleaned) is
    # now a blank rectangle. Leaving it in the layout would draw that blank
    # on top of the shape and hide it — exactly the page-15 "01/02/03
    # badges go missing" symptom.
    shape_boxes = (
        [_box4(r) for r in subicon_records]
        + [_box4(r) for r in line_subicon_records]
        + [_box4(r) for r in internal_shape_records]
    )

    def _covered_by_shape(fx1, fy1, fx2, fy2) -> bool:
        farea = max(1, (fx2 - fx1) * (fy2 - fy1))
        for sx1, sy1, sx2, sy2 in shape_boxes:
            ox1, oy1 = max(fx1, sx1), max(fy1, sy1)
            ox2, oy2 = min(fx2, sx2), min(fy2, sy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            if (ox2 - ox1) * (oy2 - oy1) >= 0.8 * farea:
                return True
        return False

    foreground_records = [
        r for r in foreground_records if not _covered_by_shape(*r)
    ]

    def _outline_duplicated_by_full_record(ox1: int, oy1: int,
                                           ox2: int, oy2: int) -> bool:
        """Drop outline masks when an equivalent full crop already exists.

        Filled cards/panels are often detected twice: once as an outline
        alpha mask and once as a near-identical foreground crop. Keeping
        both makes PowerPoint stacks noisy and can visually double borders
        or residual pixels. Prefer the full crop for same-sized duplicates
        because it carries the panel fill/background in one object.
        """
        oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
        for fx1, fy1, fx2, fy2 in background_records + foreground_records:
            farea = max(1, (fx2 - fx1) * (fy2 - fy1))
            size_ratio = min(oarea, farea) / max(oarea, farea)
            if size_ratio < 0.80:
                continue
            ix1, iy1 = max(ox1, fx1), max(oy1, fy1)
            ix2, iy2 = min(ox2, fx2), min(oy2, fy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.92 * min(oarea, farea):
                return True
        return False

    outline_records = [
        r for r in outline_records
        if not _outline_duplicated_by_full_record(r[0], r[1], r[2], r[3])
    ]

    # Nested-foreground inpaint pass. When a small foreground record (a
    # brand icon, an empty-circle inner glyph, a bullet, a connector tip)
    # sits fully inside a larger foreground record (the card / row /
    # panel that visually contains it), BOTH get cropped from the cleaned
    # image — so the parent's asset still carries the icon, and moving or
    # deleting the child element in PPT reveals an identical ghost icon
    # underneath. The subicon and internal-shape paths only cover
    # white-on-dark cards and dense numbered badges; this handles the
    # general case (line-art icons on white cards, brand logos in title
    # bars, etc.).
    #
    # Crucially we also track which children got inpainted into a parent:
    # their own asset crop must NOT come from the now-erased cleaned
    # image, or the icon vanishes from the rendered deck while only the
    # surrounding bg shows (the page-4 "empty circle, missing icon
    # inside" symptom). Inpainted children get re-routed to the
    # text-only sidecar — strokes intact, OCR text already erased.
    inpainted_children: set[tuple[int, int, int, int]] = set()

    def _inpaint_nested_foreground_in_parents() -> None:
        if not foreground_records:
            return
        # Pre-compute child fg masks BEFORE mutating cleaned. If the same
        # icon nests inside multiple parents we want every parent
        # inpainted from the SAME source mask.
        child_min_side = s_length(4, scale)
        child_max_area = s_area(18000, scale)
        child_max_side = s_length(180, scale)
        child_min_fg_pixels = s_area(20, scale)
        child_masks: list[tuple] = []
        for child in foreground_records:
            cx1, cy1, cx2, cy2 = child
            cw, ch = cx2 - cx1, cy2 - cy1
            if cw <= child_min_side or ch <= child_min_side:
                continue
            c_area = cw * ch
            # Only treat small bboxes as icons-to-erase. ~135×135 cap
            # (at 720-scale) covers brand logos, bullet glyphs, connector
            # tips, and the inner glyphs of card icons; leaves cards
            # /panels alone so they stay in the parent's bg.
            if c_area > child_max_area or max(cw, ch) > child_max_side:
                continue
            crop = cleaned[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            local_fg = (gray < 245) | (hsv[:, :, 1] > 12)
            if int(local_fg.sum()) < child_min_fg_pixels:
                continue
            child_masks.append((child, c_area, local_fg))

        for child, c_area, local_fg in child_masks:
            cx1, cy1, cx2, cy2 = child
            for parent in foreground_records:
                if parent == child:
                    continue
                px1, py1, px2, py2 = parent
                p_area = (px2 - px1) * (py2 - py1)
                # Parent must be meaningfully bigger so two near-equal
                # icons don't get paired. 2× catches icons sitting in
                # their tight bounding card.
                if p_area <= c_area * 2:
                    continue
                ox1, oy1 = max(cx1, px1), max(cy1, py1)
                ox2, oy2 = min(cx2, px2), min(cy2, py2)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                if (ox2 - ox1) * (oy2 - oy1) < 0.85 * c_area:
                    continue
                parent_local = cleaned[py1:py2, px1:px2]
                ph, pw = parent_local.shape[:2]
                # Translate the child's local fg mask into the parent's
                # crop frame and inpaint. Clip the destination in case
                # the child slips past the parent (containment above is
                # area-based, not strict).
                pm = np.zeros((ph, pw), dtype=bool)
                ty1 = cy1 - py1
                tx1 = cx1 - px1
                dy1, dx1 = max(0, ty1), max(0, tx1)
                dy2 = min(ph, ty1 + local_fg.shape[0])
                dx2 = min(pw, tx1 + local_fg.shape[1])
                if dy2 <= dy1 or dx2 <= dx1:
                    continue
                sy1, sx1 = dy1 - ty1, dx1 - tx1
                sy2 = sy1 + (dy2 - dy1)
                sx2 = sx1 + (dx2 - dx1)
                pm[dy1:dy2, dx1:dx2] = local_fg[sy1:sy2, sx1:sx2]
                inpaint_region_inplace(parent_local, pm, scale=scale)
                inpainted_children.add(tuple(child))

    _inpaint_nested_foreground_in_parents()

    # Coverage safety net: anything visible in the original cleaned image
    # that doesn't fall under SOME inventory bbox (foreground, subicon,
    # internal, outline) becomes a fallback foreground. Catches isolated
    # decorations that all detectors missed — guarantees no visible pixel
    # silently disappears in the output. Read the FG mask from the text-
    # only sidecar (saved at start of main) so shape inpainting doesn't
    # mask the residual check.
    text_only_for_residual = cv2.imread(str(text_only_path))
    if text_only_for_residual is not None:
        residual_mask = _foreground_mask(text_only_for_residual, args.dilate)
        all_bboxes = (
            list(foreground_records)
            + [_box4(r) for r in subicon_records]
            + [_box4(r) for r in line_subicon_records]
            + [_box4(r) for r in internal_shape_records]
            + [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in outline_records]
        )
        covered = np.zeros_like(residual_mask, dtype=np.uint8)
        for bx1, by1, bx2, by2 in all_bboxes:
            covered[by1:by2, bx1:bx2] = 255
        leftover = cv2.bitwise_and(residual_mask, cv2.bitwise_not(covered))
        ln, _, lstats, _ = cv2.connectedComponentsWithStats(leftover, 8)
        residual_min_area = s_area(400, scale)
        residual_min_side = s_length(18, scale)
        for i in range(1, ln):
            lx, ly, lw, lh, larea = lstats[i]
            if larea < residual_min_area:
                continue
            # Avoid creating duplicate-sized records that hug an existing
            # outline edge. Require at least one dimension to look like a
            # real element (≥18 px at 720-scale).
            if lw < residual_min_side and lh < residual_min_side:
                continue
            foreground_records.append((int(lx), int(ly),
                                       int(lx + lw), int(ly + lh)))

    # Backgrounds first (so they sort to the back via y-position tie-break),
    # then foregrounds.
    for x1, y1, x2, y2 in background_records:
        inventory.append({
            "id": f"v{visual_idx:03d}",
            "type": "image",
            "bbox": [x1, y1, x2, y2],
            "source": "cleaned",
            "role": "background",
        })
        visual_idx += 1
    masks_out_dir = Path(args.masks_dir) if args.masks_dir else None
    if outline_records and masks_out_dir is None:
        out_path = Path(args.out)
        masks_out_dir = out_path.with_name(f"{out_path.stem}_masks")
    if masks_out_dir is not None:
        masks_out_dir.mkdir(parents=True, exist_ok=True)
    # Outlines emit before foreground records so the rounded border
    # renders behind the interior content via the layout sort key below.
    for x1, y1, x2, y2, outline_mask in outline_records:
        comp_id = f"v{visual_idx:03d}"
        entry = {
            "id": comp_id,
            "type": "image",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "source": "cleaned",
            "role": "outline",
        }
        if masks_out_dir is not None:
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = outline_mask
            mask_path = masks_out_dir / f"{comp_id}.mask.png"
            cv2.imwrite(str(mask_path), full_mask)
            entry["mask_path"] = str(mask_path)
        inventory.append(entry)
        visual_idx += 1
    for x1, y1, x2, y2 in foreground_records:
        comp_id = f"v{visual_idx:03d}"
        # Inpainted nested children must crop from the text-only sidecar
        # (strokes intact); other foregrounds crop from cleaned where
        # shape inpainting has already removed embedded children.
        is_nested_child = (x1, y1, x2, y2) in inpainted_children
        entry = {
            "id": comp_id,
            "type": "image",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "source": (
                "source" if (args.use_fastsam or is_nested_child) else "cleaned"
            ),
        }
        # When FastSAM provided the segment, persist its mask alongside
        # the inventory so the asset cropper can alpha-key it.
        mask = foreground_to_mask.get((x1, y1, x2, y2))
        if mask is not None and masks_out_dir is not None:
            mask_path = masks_out_dir / f"{comp_id}.mask.png"
            cv2.imwrite(str(mask_path), mask)
            entry["mask_path"] = str(mask_path)
        inventory.append(entry)
        visual_idx += 1
    # Subicons last: they have to render IN FRONT of their parent card so the
    # white icon is visible. The layout sort by (y, x) then by element type
    # in inventory_to_layout keeps text on top, images behind — within the
    # image group, subicons appear after their parent because they're
    # appended after foreground_records.
    for x1, y1, x2, y2 in internal_shape_records:
        inventory.append({
            "id": f"v{visual_idx:03d}",
            "type": "image",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "source": "source",
            "role": "internal",
        })
        visual_idx += 1
    for x1, y1, x2, y2, line_mask in line_subicon_records:
        comp_id = f"v{visual_idx:03d}"
        entry = {
            "id": comp_id,
            "type": "image",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "source": "source",
            "role": "line_subicon",
        }
        if masks_out_dir is not None:
            mask_path = masks_out_dir / f"{comp_id}.mask.png"
            cv2.imwrite(str(mask_path), line_mask)
            entry["mask_path"] = str(mask_path)
        inventory.append(entry)
        visual_idx += 1
    for x1, y1, x2, y2 in subicon_records:
        inventory.append({
            "id": f"v{visual_idx:03d}",
            "type": "image",
            "bbox": [x1, y1, x2, y2],
            "source": "source",
            "role": "subicon",
        })
        visual_idx += 1

    def _inventory_sort_key(e: dict) -> tuple:
        # Keep extracted child visuals above their parent crops, and
        # ensure card outlines render behind the foreground interiors
        # they wrap. A plain y/x sort can put a leftmost internal icon
        # before the large parent that contains it; the parent then
        # paints over the icon in PowerPoint.
        role_order = {
            "background": 0,
            "outline": 2,
            "internal": 3,
            "line_subicon": 4,
            "subicon": 4,
        }
        if e.get("type") == "image":
            role = e.get("role")
            return (0, role_order.get(role, 1), e["bbox"][1], e["bbox"][0])
        return (1, 0, e["bbox"][1], e["bbox"][0])

    inventory.sort(key=_inventory_sort_key)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    # Persist the icon-filled cleaned image so the parent's asset crop
    # (taken from the cleaned image in inventory_to_layout.py) comes out
    # without the embedded white pictogram. Overwriting the same path keeps
    # the pipeline plumbing simple — the cleaned image is "the latest view".
    cv2.imwrite(args.clean, cleaned)

    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.source).stem
        vis = source.copy()
        # Background components — gray
        for x1, y1, x2, y2 in background_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)
        # Outline-only card borders — yellow
        for x1, y1, x2, y2, _outline_mask in outline_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 1)
        # Foreground (parent image) components — blue
        for x1, y1, x2, y2 in foreground_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 1)
        # OCR-derived editable text — green
        for it in candidate_texts:
            cv2.rectangle(vis, (it["x1"], it["y1"]), (it["x2"], it["y2"]),
                          (0, 200, 0), 1)
        # Sub-icons (white-on-dark extracts) — magenta, thicker
        for x1, y1, x2, y2 in subicon_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(vis, "sub", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1,
                        cv2.LINE_AA)
        # Sparse line-art sub-icons — purple, thicker
        for x1, y1, x2, y2 in [_box4(r) for r in line_subicon_records]:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 0, 180), 2)
            cv2.putText(vis, "line", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 180), 1,
                        cv2.LINE_AA)
        # Internal shapes (badges/placeholders extracts) — orange, thicker
        for x1, y1, x2, y2 in internal_shape_records:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(vis, "int", (x1 + 1, max(10, y1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 255), 1,
                        cv2.LINE_AA)
        # Legend in top-left
        legend = [
            ("foreground (img element)", (255, 100, 0)),
            ("background (whole-slide)", (160, 160, 160)),
            ("outline (card border)", (0, 220, 220)),
            ("editable text", (0, 200, 0)),
            ("subicon (movable)", (255, 0, 255)),
            ("line subicon (movable)", (180, 0, 180)),
            ("internal shape (movable)", (0, 100, 255)),
        ]
        for li, (label, col) in enumerate(legend):
            cv2.rectangle(vis, (8, 8 + li * 18), (24, 22 + li * 18), col, -1)
            cv2.putText(vis, label, (28, 20 + li * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                        cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_inv.png"), vis)

        # _cv2.png — shows the raw foreground mask cv2 detect_components
        # uses, plus every connected component's bbox (before any
        # subicon/internal-shape filtering). Useful for diagnosing
        # "missing card border" cases: if the mask is empty where a card
        # should be, the threshold (gray<245 OR S>12) didn't pick up its
        # background colour.
        cv2_mask = _foreground_mask(cleaned, args.dilate)
        # Show source ghosted at 40% under the mask in red.
        vis2 = source.copy()
        mask_red = np.zeros_like(vis2)
        mask_red[..., 2] = cv2_mask  # red where foreground
        vis2 = cv2.addWeighted(vis2, 0.4, mask_red, 0.6, 0)
        # Overlay every raw connected-component bbox in cyan (these are
        # what detect_components returns before any later filtering).
        n_cv, _, stats_cv, _ = cv2.connectedComponentsWithStats(cv2_mask, 8)
        kept = 0
        for i in range(1, n_cv):
            x, y, ww, hh, area = stats_cv[i]
            if area < args.min_area:
                continue
            cv2.rectangle(vis2, (int(x), int(y)),
                          (int(x + ww), int(y + hh)),
                          (200, 200, 0), 1)
            kept += 1
        cv2.putText(vis2, f"{kept} components (min_area={args.min_area})",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(debug_dir / f"{stem}_cv2.png"), vis2)

    text_count = sum(1 for e in inventory if e["type"] == "text")
    img_count = sum(1 for e in inventory if e["type"] == "image")
    sub_count = sum(1 for e in inventory if e.get("role") == "subicon")
    line_count = sum(1 for e in inventory if e.get("role") == "line_subicon")
    print(json.dumps({"text": text_count, "image": img_count,
                      "subicon": sub_count, "line_subicon": line_count,
                      "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
