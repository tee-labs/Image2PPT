#!/usr/bin/env python
"""Convert an inventory JSON into asset manifest + layout JSON.

For each inventory entry:
  - "text":  → text element in layout. Auto-detect color (sampling from
    the source image), bold (stroke-density ratio), and font size
    (bbox height x type-aware multiplier).
  - "image": → asset manifest crop spec + image element in layout.

Z-order: image elements come FIRST in layout (rendered behind), text
elements come LAST (rendered on top, never covered by images). All
text shapes get transparent fill via build_pptx_from_layout.py.

Font size formula (type-aware multiplier on bbox height):
  - Big stat values (digits + unit, e.g. "20.26%", "29 PB"): 0.68x
  - Short labels (<=4 chars):                                0.58x
  - Medium text (5-12 chars):                                0.55x
  - Long text / paragraphs:                                  0.52x

File layout (search the section banners):
  - CLI: parse_args
  - Containment topo sort (parents render before children)
  - Text style detection: color / bold / font size
  - Geometry helpers: hex/RGB, overlap, intersect
  - Icon bbox padding + short-stat trim
  - Group unification + multi-line text merge
  - Main: build manifest + layout, crop asset PNGs

Usage:
    python inventory_to_layout.py --inventory inv.json --source slide.png \\
        --asset-prefix assets/page_001 \\
        --cleaned clean.png \\
        --out-manifest m.json --out-layout l.json --out-assets-dir assets/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fontconfig_helper import fontconfig_font_path  # noqa: E402
from icon import inpaint_region_inplace  # noqa: E402
from text_safety import ppt_safe_text  # noqa: E402


ENABLE_NATIVE_OUTLINE_SHAPES = False

_FONT_CANDIDATES_CACHE: list[dict] | None = None
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_TEXT_RENDER_CACHE: dict[tuple[str, str, int, bool], dict | None] = {}


def default_ppt_font() -> str:
    """Target PPT font family.

    Local QA on macOS may render this through a substitute if YaHei is not
    installed, but the PPTX itself should preserve the intended family.
    """
    return "Microsoft YaHei"


def _fontconfig_font_path(query: str) -> str | None:
    """Resolve a font family through fontconfig when available."""
    return fontconfig_font_path(query)


def estimate_slide_background_hex(img: np.ndarray) -> str:
    """Estimate the slide canvas colour from corner patches.

    Generated slide screenshots usually have a solid canvas behind the
    content. Sampling corners keeps dark themes dark in the rebuilt PPTX
    and avoids painting letterbox margins white for portrait/square pages.
    """
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
        return "#FFFFFF"
    quant = (pixels // 16) * 16
    from collections import Counter
    winner, _ = Counter(map(tuple, quant)).most_common(1)[0]
    winner_arr = np.array(winner, dtype=np.int16)
    diff = np.abs(pixels.astype(np.int16) - winner_arr).max(axis=1)
    close = pixels[diff <= 24]
    color = np.median(close if len(close) else pixels, axis=0).astype(int)
    b, g, r = (int(v) for v in color)
    return f"#{r:02X}{g:02X}{b:02X}"


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert inventory to manifest + layout.")
    parser.add_argument("--inventory", required=True, help="Inventory JSON path.")
    parser.add_argument("--source", required=True, help="Original source image (for style detection).")
    parser.add_argument("--cleaned", required=True, help="Cleaned image (for asset crops).")
    parser.add_argument("--asset-prefix", required=True,
                        help="Asset path prefix in layout (e.g. 'assets/page_001').")
    parser.add_argument("--out-assets-dir", required=True, help="Directory to save asset PNGs.")
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-layout", required=True)
    parser.add_argument(
        "--slide-width-in", type=float, default=None,
        help="Slide width in inches. Default: auto-compute from "
             "source aspect ratio so 4:3 inputs get a 4:3 slide and "
             "16:9 inputs get 13.333 × 7.5.",
    )
    parser.add_argument("--slide-height-in", type=float, default=7.5)
    return parser.parse_args()


# =============================================================================
# Z-order: parents before children
# =============================================================================


def topo_sort_by_containment(
    elements: list[dict],
    containment: float = 0.85,
) -> list[dict]:
    """Reorder image elements so visual parents render before their children.

    Builds a DAG where `parent → child` whenever `parent.bbox` is strictly
    larger than `child.bbox` AND covers at least `containment` (default
    85 %) of the child's area. A Kahn-style topological sort then emits
    every parent before any of its descendants, so PowerPoint paints
    parents first (back) and children last (front). Without this, a
    parent card can land *above* an icon nested inside it and hide it.

    The strict `parent_area > child_area` requirement prevents cycles
    between two roughly equal-size overlapping boxes (neither parents
    the other); the threshold prevents incidental edge-touches from
    creating spurious parent links.

    Tiebreaker for nodes with no remaining parents: larger area first
    (the natural back), then top-to-bottom, left-to-right. Falls back to
    the input order if a cycle is somehow produced (shouldn't happen
    given the strict-area rule, but the guard is cheap).
    """
    n = len(elements)
    if n < 2:
        return elements

    boxes = []
    for e in elements:
        x, y, w, h = e["box"]
        boxes.append((x, y, x + w, y + h, max(1, w * h)))

    parents_of: list[set[int]] = [set() for _ in range(n)]
    children_of: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        ix1, iy1, ix2, iy2, ia = boxes[i]
        for j in range(n):
            if i == j:
                continue
            jx1, jy1, jx2, jy2, ja = boxes[j]
            if ja <= ia:
                continue
            ox1 = max(ix1, jx1)
            oy1 = max(iy1, jy1)
            ox2 = min(ix2, jx2)
            oy2 = min(iy2, jy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            inter = (ox2 - ox1) * (oy2 - oy1)
            if inter >= containment * ia:
                parents_of[i].add(j)
                children_of[j].add(i)

    def tiebreak(i: int) -> tuple:
        x, y, _, _, area = boxes[i]
        return (-area, y, x, i)

    indeg = [len(p) for p in parents_of]
    available = sorted([i for i in range(n) if indeg[i] == 0], key=tiebreak)

    result: list[dict] = []
    while available:
        i = available.pop(0)
        result.append(elements[i])
        for c in children_of[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ck = tiebreak(c)
                lo = 0
                hi = len(available)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tiebreak(available[mid]) <= ck:
                        lo = mid + 1
                    else:
                        hi = mid
                available.insert(lo, c)

    if len(result) != n:
        return elements
    return result


def find_title_parent(
    text_bbox: tuple[int, int, int, int],
    candidates: list[dict],
    tol_frac: float = 0.05,
) -> tuple[tuple[int, int, int, int], bool] | None:
    """Detect a centred-title relationship between an OCR text and an image.

    Catches nested outlines that don't survive into the rendered layout
    (e.g. a stat sub-card inside a bigger card). The build-time pass in
    build_pptx_from_layout.apply_title_centering catches the complementary
    case — non-outline images (per-card source crops with no role tag).
    Both passes are idempotent on already-centred boxes.

    Returns ``(parent_bbox, y_centered)`` when the text is strictly
    contained in a candidate, the smallest such candidate's x-centre
    matches the text x-centre within ``tol_frac`` × parent width, and
    optionally tells whether y also matches. Smallest area wins so
    nested cards take priority over full-slide backgrounds.
    """
    tx1, ty1, tx2, ty2 = text_bbox
    tcx = (tx1 + tx2) / 2.0
    tcy = (ty1 + ty2) / 2.0
    # Filter to matching candidates first, then pick the smallest among
    # them. Doing it the other way round (smallest first, then check) can
    # silently drop a valid larger parent when a smaller intermediate
    # container fails the centre test — e.g. a slightly off-centre inner
    # card inside a perfectly-aligned outer one.
    best: tuple[int, int, int, int] | None = None
    best_area: int | None = None
    best_y_centered = False
    for el in candidates:
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        px1, py1, px2, py2 = (int(v) for v in bbox)
        if not (px1 < tx1 and py1 < ty1 and px2 > tx2 and py2 > ty2):
            continue
        tol = tol_frac * (px2 - px1)
        if abs(tcx - (px1 + px2) / 2.0) > tol:
            continue
        area = (px2 - px1) * (py2 - py1)
        if best_area is None or area < best_area:
            best = (px1, py1, px2, py2)
            best_area = area
            best_y_centered = abs(tcy - (py1 + py2) / 2.0) <= tol
    if best is None:
        return None
    return best, best_y_centered


# =============================================================================
# Text style detection: color / bold / font size
# =============================================================================


def _classify_text_color(r: int, g: int, b: int, bg: np.ndarray) -> str:
    """Map a sampled text-stroke (r,g,b) → normalized hex color.

    Extracted from detect_text_style so per-character sampling can reuse
    the same color-classification logic at the line level. The
    classification matters: anti-aliased samples within ±10 of pure
    black often land at e.g. (18, 32, 28) — without normalization to
    `#222222`, two visually-identical black chars produce different
    hex strings and break run-grouping below.
    """
    bg_int = bg.astype(int)
    bg_lum = int(0.114 * bg_int[0] + 0.587 * bg_int[1] + 0.299 * bg_int[2])
    bg_is_dark = bg_lum < 110
    text_lum = int(0.114 * b + 0.587 * g + 0.299 * r)
    if bg_is_dark and text_lum > 170:
        return "#FFFFFF"
    if r > g + 30 and r > b + 30 and r > 150:
        return "#D43E3E"
    # Canonical deck blue. Require a real blue-vs-gray separation; low
    # saturation grey-blue body text such as #7A8191 should remain grey
    # instead of being snapped to the saturated title blue.
    if b > r + 28 and b > g + 18 and b > 100:
        return "#054798"
    if r < 80 and g < 80 and b < 80:
        return "#222222"
    # Anti-aliased near-black: per-char sampling on a thin stroke can
    # land at e.g. (92, 92, 93) — visually identical to (34, 34, 34)
    # but missed by the strict r<80,g<80,b<80 gate. Catch low-saturation
    # dark grays by combining max-channel and per-channel spread. Without
    # this, in-bbox color sampling over-splits gray sentences into many
    # separate runs (`医`#222222 / `疗`#5C5C5D / `政务`#222222 / ...).
    max_ch = max(r, g, b)
    min_ch = min(r, g, b)
    # Two-tier near-black: very-low-saturation samples up to max_ch=130
    # still read as black-ish on light backgrounds — per-char sampling on
    # thin strokes routinely lands at (108,108,110) or (122,125,125) on
    # what's visually solid dark text. Without absorbing these into
    # #222222, run-grouping splits a single-color sentence into
    # alternating chunks of #6C6C6E / #222222 / #727375 / ...
    if max_ch < 130 and (max_ch - min_ch) < 12:
        return "#222222"
    return f"#{r:02X}{g:02X}{b:02X}"


def _per_char_runs(bbox: list[int], text: str, orig_img: np.ndarray,
                   bg: np.ndarray, fallback_color: str,
                   char_boxes: list[list[int]] | None = None,
                   words: list[str] | None = None,
                   word_boxes: list[list[int]] | None = None) -> list[dict]:
    """Detect in-bbox color variation by sampling each character/word and
    grouping consecutive same-colored segments into runs.

    PaddleOCR returns one bbox per text line; when a line has mixed
    colors, the line-level median washes the variation to one dominant
    color. This function recovers it.

    Three sampling modes (preference order):

      C) Per-WORD bboxes from PaddleOCR (`words` + `word_boxes` from
         return_word_box=True). PP-OCRv5 segments by character class
         (continuous digits as one word, each CJK glyph as one word),
         so a `524个` record arrives as words=['524','个'] with two
         distinct bboxes — exactly the granularity at which colour
         changes typically happen. Sampled per-word; word boundaries
         are trusted, so the per-char single-char-suppression rule
         (which would fold a single `个` back into the dominant red)
         is bypassed. Used when len(words) ≥ 2.

      A) Real per-CHAR bboxes (`char_boxes`). Used for pure-CJK lines
         where PaddleOCR returns one box per glyph. char_boxes are
         only attached by ocr_paddle when each word is a single char
         (i.e. no digit clusters), so this branch handles long CJK
         sentences with embedded colour emphasis.

      B) Proportional width estimation (fallback). When neither real
         word nor char boxes are available (or alignment fails after
         OCR-review text correction), allocate each character a width
         unit (CJK 1.0, ASCII 0.55, space 0.4, newline 0) and slice
         the line bbox by proportion.

    In each char slice, pixels far from `bg` (the same ring-sampled
    background used at the line level) are text strokes; their median
    gives the char color. Consecutive chars within ±25 per channel are
    grouped into one run.

    Returns a list of {"text": str, "color": "#RRGGBB"} dicts. The
    caller decides whether to emit a `runs` field (only when len > 1).
    """
    x1, y1, x2, y2 = bbox
    region = orig_img[y1:y2, x1:x2]
    bbox_w = x2 - x1
    if region.size == 0 or bbox_w <= 0 or len(text) <= 1:
        return [{"text": text, "color": fallback_color}]

    # Mode C: per-WORD bboxes (preferred when ≥ 2 words). Trusts
    # PaddleOCR's character-class segmentation as the natural unit for
    # in-line colour AND size variation. PP-OCRv5 returns the same y
    # range for every word on a line (inherited from the line bbox),
    # so per-word font-size CAN'T be derived from word_box height
    # alone. Instead, threshold ink pixels inside each word_box and
    # measure their actual vertical extent — that's the true glyph
    # height for `524` vs `个` in `524个` where the digits are bigger
    # and the unit suffix is smaller.
    def _has_digit_or_ascii(s: str) -> bool:
        return any((c.isascii() and (c.isalnum() or c in "+-.%/"))
                   for c in s)

    def _has_cjk(s: str) -> bool:
        return any(c >= "　" and not c.isascii() for c in s)

    mixed_class = _has_digit_or_ascii(text) and _has_cjk(text)
    if (words is not None and word_boxes is not None
            and len(words) == len(word_boxes)
            and len(words) >= 2
            and "".join(words) == text
            and mixed_class):
        # Mode C targets MIXED-CLASS lines (digit/ASCII cluster + CJK
        # part: `524个`, `7分钟`, `20.26%`, `《...2025年...》`). On
        # these, per-segment colour/size differences are typical and
        # PaddleOCR's word_box boundaries align with the visual change.
        # Pure CJK lines (`推动数据要素市场化配置改革。`) have each
        # glyph as a one-char word — Mode C there would devolve into
        # noisy per-char colour/size from antialiasing. Fall through
        # to Mode A on those.
        h_img, w_img = orig_img.shape[:2]
        # PP-OCRv5 occasionally returns overlapping word_boxes (e.g. a
        # `分` box extending into the next `钟`'s ink). Clip each box's
        # right edge to the next word's start x and the left edge to
        # the previous word's end so samples capture only the intended
        # glyph.
        clipped: list[tuple[int, int, int, int]] = []
        for i, wb in enumerate(word_boxes):
            wx1 = int(wb[0]); wy1 = int(wb[1])
            wx2 = int(wb[2]); wy2 = int(wb[3])
            if i > 0:
                wx1 = max(wx1, int(word_boxes[i - 1][2]))
            if i + 1 < len(word_boxes):
                wx2 = min(wx2, int(word_boxes[i + 1][0]))
            clipped.append((max(0, min(w_img, wx1)),
                            max(0, min(h_img, wy1)),
                            max(0, min(w_img, wx2)),
                            max(0, min(h_img, wy2))))
        # First pass: sample colour + glyph height per word.
        per_word: list[dict] = []
        for w, (wx1, wy1, wx2, wy2) in zip(words, clipped):
            if wx2 <= wx1 or wy2 <= wy1:
                per_word.append({"text": w, "color": fallback_color,
                                 "glyph_h": None})
                continue
            sub = orig_img[wy1:wy2, wx1:wx2]
            col = fallback_color
            glyph_h: int | None = None
            if sub.size > 0:
                diff = np.abs(sub.astype(int) - bg).max(axis=2)
                mask = diff > 35
                if int(mask.sum()) >= 3:
                    # Colour: median of strokes, snapped to dominant if close.
                    m = np.median(sub[mask], axis=0).astype(int)
                    raw = f"#{int(m[2]):02X}{int(m[1]):02X}{int(m[0]):02X}"
                    col = (fallback_color
                           if _color_close(raw, fallback_color, tol=60)
                           else raw)
                    # Glyph height: vertical extent of stroke rows that
                    # actually contain ink. Filtering by per-row count ≥ 2
                    # rejects single antialias pixels that would otherwise
                    # extend the range to the full bbox.
                    row_has_ink = mask.sum(axis=1) >= 2
                    ink_rows = np.where(row_has_ink)[0]
                    if ink_rows.size >= 2:
                        glyph_h = int(ink_rows.max() - ink_rows.min() + 1)
            per_word.append({"text": w, "color": col, "glyph_h": glyph_h})
        # Two-pass merge:
        #   1. Snap each entry's raw sampled colour to whichever
        #      cluster centroid (running list) it's within ±40 channels
        #      of. Anti-aliasing makes adjacent CJK chars in the same
        #      visual colour sample to slightly different hex (`#0B206F`
        #      vs `#10286F`) — exact-equality merging fails on these.
        #   2. Merge consecutive entries with same clustered colour AND
        #      glyph height within 30 %. Different-size segments stay
        #      separate (`7分钟`: `7` tall, `分钟` small → 2 runs).
        clusters: list[str] = []
        for entry in per_word:
            raw_col = entry["color"]
            matched = None
            for c in clusters:
                if _color_close(raw_col, c, tol=40):
                    matched = c
                    break
            if matched is None:
                clusters.append(raw_col)
                matched = raw_col
            entry["color"] = matched
        word_runs: list[dict] = []
        for entry in per_word:
            if word_runs:
                prev = word_runs[-1]
                prev_h = prev.get("glyph_h")
                cur_h = entry["glyph_h"]
                heights_close = (
                    prev_h is None or cur_h is None
                    or max(prev_h, cur_h) <= 1.3 * min(prev_h, cur_h)
                )
                if prev["color"] == entry["color"] and heights_close:
                    prev["text"] += entry["text"]
                    if (cur_h is not None
                            and (prev_h is None or cur_h > prev_h)):
                        prev["glyph_h"] = cur_h
                    continue
            word_runs.append(dict(entry))
        return word_runs

    char_colors: list[tuple[str, str | None]] = []

    # Mode A: real PaddleOCR per-char bboxes
    use_real = (
        char_boxes is not None
        and len(char_boxes) == len(text)
    )
    if use_real:
        h_img, w_img = orig_img.shape[:2]
        for c, cb in zip(text, char_boxes):
            if c == "\n":
                char_colors.append((c, None))
                continue
            # Char bbox is in source-image coords (same frame as line bbox).
            cx1 = max(0, int(cb[0]))
            cy1 = max(0, int(cb[1]))
            cx2 = min(w_img, int(cb[2]))
            cy2 = min(h_img, int(cb[3]))
            # Inset by 1 px on each side to avoid anti-aliased edge bleed
            # between adjacent characters — small fixed inset, since CJK
            # char boxes from PP-OCRv5 are typically 20-40 px wide.
            inset = 1
            cx1i = cx1 + inset
            cy1i = cy1 + inset
            cx2i = max(cx1i + 1, cx2 - inset)
            cy2i = max(cy1i + 1, cy2 - inset)
            sub = orig_img[cy1i:cy2i, cx1i:cx2i]
            if sub.size == 0:
                char_colors.append((c, None))
                continue
            diff = np.abs(sub.astype(int) - bg).max(axis=2)
            mask = diff > 35
            if int(mask.sum()) < 3:
                char_colors.append((c, None))
                continue
            pixels = sub[mask]
            m = np.median(pixels, axis=0).astype(int)
            col = f"#{int(m[2]):02X}{int(m[1]):02X}{int(m[0]):02X}"
            char_colors.append((c, col))
    else:
        # Mode B: proportional width estimation
        units = []
        total = 0.0
        for c in text:
            if c == "\n":
                u = 0.0
            elif c == " ":
                u = 0.4
            elif c.isascii() or c in ".,:;()[]{}-+*/=!?\"'":
                u = 0.55
            else:
                u = 1.0
            units.append(u)
            total += u
        if total <= 0:
            return [{"text": text, "color": fallback_color}]
        cur = 0.0
        for c, u in zip(text, units):
            if u == 0:
                char_colors.append((c, None))
                continue
            cx_start = int(cur / total * bbox_w)
            cur += u
            cx_end = int(cur / total * bbox_w)
            slice_w = cx_end - cx_start
            if slice_w < 2:
                char_colors.append((c, None))
                continue
            margin = max(1, slice_w // 5)
            sx1, sx2 = cx_start + margin, max(cx_start + margin + 1, cx_end - margin)
            sub = region[:, sx1:sx2]
            if sub.size == 0:
                char_colors.append((c, None))
                continue
            diff = np.abs(sub.astype(int) - bg).max(axis=2)
            mask = diff > 35
            if int(mask.sum()) < 3:
                char_colors.append((c, None))
                continue
            pixels = sub[mask]
            m = np.median(pixels, axis=0).astype(int)
            col = f"#{int(m[2]):02X}{int(m[1]):02X}{int(m[0]):02X}"
            char_colors.append((c, col))

    # Anchor on the line-level dominant color: any per-char sample close
    # to dominant gets normalized to dominant. This prevents anti-aliasing
    # + stroke-density noise (e.g. gray sampled as #B3B3B5 vs #A7A9A9 vs
    # #CACDD0 on three adjacent chars of the same visually-gray line)
    # from spuriously splitting the line into runs. Real color jumps
    # (red vs gray = 100+ channel diff) still cross this gate cleanly.
    #
    # Two-tier closeness:
    #   - Standard: ±40 per channel — handles most anti-aliasing variance.
    #   - Gray-aware: when BOTH dominant and sampled are near-neutral
    #     (R/G/B within ±20 of each other on each side), the apparent
    #     luminance swing from per-character stroke density is large
    #     (#222222 vs #5C5C5D for one denser glyph in a gray line). Allow
    #     up to 70 luminance difference for two near-greys before
    #     treating it as a real color jump.
    fr, fg, fb = _hex_to_rgb(fallback_color)
    fallback_is_gray = (abs(fr - fg) < 20 and abs(fg - fb) < 20
                       and abs(fr - fb) < 20)
    fallback_lum = 0.299 * fr + 0.587 * fg + 0.114 * fb
    for i, (c, col) in enumerate(char_colors):
        if col is None or col == fallback_color:
            continue
        # ±60 tolerance for "snap to line dominant color". This is
        # generous enough to absorb anti-aliased + slight-tint variance
        # (raw "#3C5F82" with a hint of blue snaps back to a gray
        # fallback like "#6A707C", max channel diff 46) while still
        # keeping real color jumps (red vs gray, channel diff > 100)
        # on separate runs.
        if _color_close(col, fallback_color, tol=60):
            char_colors[i] = (c, fallback_color)
            continue
        if fallback_is_gray:
            cr, cg, cb = _hex_to_rgb(col)
            col_is_gray = (abs(cr - cg) < 20 and abs(cg - cb) < 20
                          and abs(cr - cb) < 20)
            if col_is_gray:
                # Per-char stroke-density swings can produce luminance
                # differences up to ~130 on the same visually-gray line:
                # a dense glyph like 道 samples at lum ~30 (dark median
                # because all pixels are stroke) while a sparse one like
                # 一 samples at lum ~160 (light median because mostly
                # bg admixture). The 130 ceiling absorbs that variance
                # without bridging real black-on-light text contrasts
                # (lum diff > 150 between true near-black #222 and true
                # light gray #DDD).
                col_lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
                if abs(col_lum - fallback_lum) < 130:
                    char_colors[i] = (c, fallback_color)

    # Group consecutive same-colored chars. ±40 fallback merges adjacent
    # deviation runs that happen to land at slightly different specific
    # hexes (e.g. two adjacent red samples #D33D3D vs #D63E3F). Chars
    # that failed sampling (None) inherit the current run.
    runs: list[dict] = []
    cur_text = ""
    cur_color: str | None = None
    for c, col in char_colors:
        if col is None:
            cur_text += c
            continue
        if (cur_color is None
                or col == cur_color
                or _color_close(col, cur_color, tol=40)):
            cur_text += c
            if cur_color is None:
                cur_color = col
        else:
            if cur_text:
                runs.append({"text": cur_text, "color": cur_color})
            cur_text = c
            cur_color = col
    if cur_text:
        runs.append({"text": cur_text, "color": cur_color or fallback_color})

    # All sampling failed → single-run fallback
    if not runs:
        return [{"text": text, "color": fallback_color}]

    # Suppress single-character non-dominant runs — they are typically
    # per-char sampling noise (e.g. one glyph happens to lean toward a
    # blue OR red tint due to a stroke artefact). True intra-line color
    # emphasis usually covers ≥ 2 visible chars (a phrase or stat
    # number like "63" or "市场化配置改革"). One-char "已" / "上" type
    # emphasis CAN happen in real designs but is rare and indistinguishable
    # from noise without further context — we err on the side of
    # suppressing the noise.
    def _visible_char_count(t: str) -> int:
        return sum(1 for c in t if c.strip())

    cleaned: list[dict] = []
    for r in runs:
        if (r["color"] != fallback_color
                and _visible_char_count(r["text"]) < 2):
            # Fold into the previous run (preferred) or start a new
            # dominant run if there is no previous one.
            if cleaned:
                cleaned[-1]["text"] += r["text"]
            else:
                cleaned.append({"text": r["text"], "color": fallback_color})
        else:
            # Merge with previous run if colors now match after folding.
            if cleaned and cleaned[-1]["color"] == r["color"]:
                cleaned[-1]["text"] += r["text"]
            else:
                cleaned.append(r)
    return cleaned


def detect_text_style(bbox: list[int], orig_img: np.ndarray,
                      text: str | None = None,
                      char_boxes: list[list[int]] | None = None,
                      words: list[str] | None = None,
                      word_boxes: list[list[int]] | None = None) -> dict:
    """Sample text region in original image; return color + bold flag.

    Sample the BACKGROUND from a 4 px ring just outside the bbox first.
    Pixels INSIDE the bbox close to the bg color (within ±35 per channel)
    are considered background; the rest are text strokes. Median of the
    text-stroke pixels gives the text color; their density gives bold.

    The previous minority-color heuristic broke for large bold glyphs whose
    strokes occupy >50 % of the bbox: it flipped them to "white-on-dark"
    even when the slide background was white. Sampling outside the bbox
    fixes that — the actual surrounding bg color is unambiguous.

    When `text` is provided, also runs per-character color sampling and,
    if multiple distinct color runs are detected in the bbox, returns
    them under the `runs` key — downstream build_pptx emits one
    python-pptx run per group so in-bbox color variation (e.g. a red
    statistic inside a gray sentence) survives into the PPTX.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = orig_img.shape[:2]
    region = orig_img[y1:y2, x1:x2]
    if region.size == 0:
        return {"color": "#054798", "bold": False}

    # Sample a 6-px ring offset 2 px from the bbox. The offset matters most
    # when OCR bboxes are tight (e.g. PaddleOCR PP-OCRv5): a 0-px gap can
    # land the inner row of the ring on character strokes, polluting the bg
    # median with text-colour pixels. The 2-px gap keeps the ring outside
    # the antialiased character edge.
    gap = 2
    ring = 6
    inner_top = max(0, y1 - gap)
    inner_bot = min(h_img, y2 + gap)
    inner_left = max(0, x1 - gap)
    inner_right = min(w_img, x2 + gap)
    bg_samples = []
    if inner_top - ring >= 0:
        bg_samples.append(orig_img[inner_top - ring:inner_top, inner_left:inner_right])
    if inner_bot + ring <= h_img:
        bg_samples.append(orig_img[inner_bot:inner_bot + ring, inner_left:inner_right])
    if inner_left - ring >= 0:
        bg_samples.append(orig_img[inner_top:inner_bot, inner_left - ring:inner_left])
    if inner_right + ring <= w_img:
        bg_samples.append(orig_img[inner_top:inner_bot, inner_right:inner_right + ring])
    if bg_samples:
        bg_pixels = np.concatenate([s.reshape(-1, 3) for s in bg_samples if s.size > 0])
        # Exclude text-color leaks: anti-aliased character strokes that
        # bleed into the ring would skew the median toward the text color.
        # Vote majority bg via quantised mode, keep only pixels close to
        # mode, then take their median. Bigger-tolerance version of the
        # _strip_bg_median helper in erase_text.py.
        quant = (bg_pixels // 16) * 16
        from collections import Counter
        mode_q = np.array(Counter(map(tuple, quant)).most_common(1)[0][0])
        diff = np.abs(bg_pixels.astype(int) - mode_q).max(axis=1)
        close = bg_pixels[diff <= 30]
        if len(close) >= max(8, int(len(bg_pixels) * 0.2)):
            bg = np.median(close, axis=0)
        else:
            bg = np.median(bg_pixels, axis=0)
    else:
        bg = np.array([255.0, 255.0, 255.0])

    # Tight-colored-container case: when an OCR bbox almost exactly fills
    # a colored badge (e.g. white `Rhombus` text on a dark-blue rounded
    # rectangle whose outer edge is within 0-2 px of the text bbox), the
    # ring sample lands OUTSIDE the badge on the slide bg and reports
    # white. The actual background of the text is the badge fill, not the
    # slide. Detect this by checking whether a clear majority of pixels
    # INSIDE the bbox are saturated and distinct from the ring color; if so,
    # use the saturated-pixel median as the effective bg.
    inner_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat_mask = inner_hsv[:, :, 1] > 50
    total_px = region.shape[0] * region.shape[1]
    # Threshold tuned to discriminate two cases:
    # - Saturated TEXT on a neutral bg (e.g. dark-blue cover title strokes
    #   on white): saturated pixels are ~20-30 % of the bbox area.
    # - Saturated CONTAINER fill with white/light text on top (e.g. white
    #   `Rhombus` on dark-blue badge): saturated pixels are ~60-80 % because
    #   the badge fill dominates.
    # 0.55 sits between the two and keeps the container case while
    # rejecting the text case.
    is_container_like = int(sat_mask.sum()) > 0.55 * total_px
    if is_container_like:
        # Disambiguate: dense CJK text (`网络运维智能体`, 7 bold glyphs in a
        # tight PP-OCRv5 bbox) can also hit 55–60 % saturated coverage
        # because the strokes fill so much of the bbox. The visual
        # difference is topology — a real container is ONE big blob, dense
        # strokes are many smaller blobs. Compute connected components of
        # the saturated mask; if the LARGEST component covers a clear
        # majority of the saturated pixels, treat it as a container, else
        # bail out and fall back to the ring-sampled bg.
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(
            sat_mask.astype(np.uint8) * 255, 8)
        if n_comp > 1:
            largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_sat = int(sat_mask.sum())
            if largest_area < 0.55 * total_sat:
                is_container_like = False
    if is_container_like:
        container_bg = np.median(region[sat_mask], axis=0)
        if int(np.max(np.abs(container_bg.astype(int) - bg.astype(int)))) > 40:
            bg = container_bg

    # Pixels within bbox far from bg are text strokes.
    diff = np.abs(region.astype(int) - bg).max(axis=2)
    text_mask = diff > 35
    if int(text_mask.sum()) < 5:
        # Fallback: take darkest 25 % vs bg if normal threshold fails
        thr = max(15, int(np.percentile(diff, 75)))
        text_mask = diff > thr
        if int(text_mask.sum()) < 5:
            return {"color": "#054798", "bold": False}

    text_pixels = region[text_mask]
    m = np.median(text_pixels, axis=0).astype(int)
    b, g, r = int(m[0]), int(m[1]), int(m[2])
    color = _classify_text_color(r, g, b, bg)
    # Raw RGB hex of the line median: keeps the same color SPACE as
    # the per-char raw samples so the anchor step inside _per_char_runs
    # can merge close-but-not-identical samples back to the line
    # dominant. Passing the CLASSIFIED color here as fallback breaks
    # the anchor: a slight-blue gray sample like #707C94 has channel
    # diff > 100 from a classified #054798, even though the underlying
    # raw line median is only ~40 channels away.
    raw_line_hex = f"#{r:02X}{g:02X}{b:02X}"

    h, w = region.shape[:2]
    # ink_h: tight vertical extent of strokes inside the bbox. Used by
    # the caller (a) as a decoration-padding probe — if a multi-char
    # line's ink_h is much less than bbox_h, the bbox likely includes
    # shadow/glow above and below the actual glyphs — and (b) as the
    # denominator for stroke-width-based bold detection below.
    ink_h: int | None = None
    row_ink = text_mask.sum(axis=1) >= 2
    rows = np.where(row_ink)[0]
    if rows.size >= 2:
        ink_h = int(rows.max() - rows.min() + 1)

    # Bold detection via absolute stroke half-width. Density
    # (ink_pixels / bbox_area) misfires across font sizes — a small
    # 12 pt body line at 0.40 density is regular, while a 32 pt bold
    # stat at 0.24 density looks identical to a regular under the
    # density rule. Stroke half-width is the cleaner signal: at
    # rasterisation distances ≥ 18 ppi, a regular CJK stroke median is
    # ≤ 1 px (the median of `[1, 1, ...]` edge-dominated values), and a
    # bold stroke median is ≥ 1.5 px. The 1.2 cutoff sits in that gap.
    # Computed on the distance transform of the ink mask — the median
    # of distances at ink pixels is a half-stroke-width proxy.
    bold = False
    if int(text_mask.sum()) >= 8 and ink_h and ink_h >= 4:
        try:
            mask_u8 = text_mask.astype(np.uint8)
            dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
            stroke_half_median = float(np.median(dist[text_mask]))
            bold = stroke_half_median > 1.2
        except cv2.error:
            density = float(text_mask.sum()) / max(1, h * w)
            bold = density > 0.27

    result = {"color": color, "bold": bool(bold), "ink_h": ink_h}

    # Per-character run detection: only meaningful when we have the text
    # AND there are at least 2 characters to compare. The runs key is
    # only added when in-bbox color variation is actually detected,
    # otherwise the single-color path stays compatible with old layouts.
    if text and len(text) >= 2:
        runs = _per_char_runs(bbox, text, orig_img, bg, raw_line_hex,
                              char_boxes=char_boxes,
                              words=words, word_boxes=word_boxes)
        if len(runs) > 1:
            result["runs"] = runs
    return result


_FONT_LADDER = [8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 28, 32,
                36, 40, 44, 48, 54, 60, 66, 72, 80, 88]


def _snap_to_ladder(pt: float) -> int:
    """Snap a font-size estimate to the nearest standard size. Without this
    snap, two visually-identical labels (e.g. `1000亿元` and `24.22%`) end
    up rendered at slightly different sizes when their OCR bbox heights
    differ by 1-2 px due to glyph-shape variation."""
    return int(min(_FONT_LADDER, key=lambda v: abs(v - pt)))


# Resolution-independent calibration constants. These are pure ratios —
# they describe how PaddleOCR's bbox relates to the underlying glyph
# extent, independent of source resolution or slide size. The
# pixel-dependent quantity is `pt_per_px = slide_h_in × 72 / source_h`,
# computed at the call site and passed in.
#
# α (CAPTURE_FACTOR_*) — bbox height in pt × α ≈ font pt. CJK glyphs
#   fill ~85 % of the em-box, and PaddleOCR pads the bbox by another
#   ~10 % vertically; the inverse of that combined factor is α ≈ 0.77.
#   Stat values are denser (digit-heavy) — OCR captures tighter, so α
#   is larger.
# β (GLYPH_WIDTH_FACTOR) — units × font_pt × β ≈ run width in pt. CJK
#   ~1.0 em wide, ASCII ~0.55 em; β=1.088 adds a small safety margin
#   against LibreOffice's slight over-rendering of wide mixed stats
#   like `7分钟`.
CAPTURE_FACTOR_SHORT = 0.78
CAPTURE_FACTOR_MEDIUM = 0.74
CAPTURE_FACTOR_LONG = 0.71


def _is_punct_run(text: str) -> bool:
    """A run is treated as punctuation when every visible char belongs
    to Unicode category P* (punctuation) or S* (symbol). Catches both
    ASCII (`+`, `"`, `,`) and CJK (`，`, `"`, `"`, `（`, `）`)
    without an explicit character list — easier to maintain and avoids
    bugs where the source-file glyph and OCR-emitted glyph use
    different code points for the same visual mark.
    """
    visible = "".join(c for c in (text or "") if not c.isspace())
    if not visible:
        return True
    import unicodedata
    return all(unicodedata.category(c)[0] in ("P", "S") for c in visible)


def _inherit_punct_sizes(runs: list[dict]) -> None:
    """Make punctuation runs follow neighbouring text's size.

    Mode C samples each PaddleOCR word_box independently. For
    punctuation like `+`, `"`, `，` the glyph itself is small (raw
    glyph_h ~8 px) so the per-word formula returns 8 pt. Typographically
    though punctuation shares the em-box with the surrounding text — a
    `"软约束"` at 16 pt should keep the quotes at 16 pt, not 8 pt.

    Rule: for each pure-punctuation run, copy the size of the nearest
    non-punctuation neighbour (prefer the larger only on a tie).
    """
    if not runs:
        return
    is_punct = [_is_punct_run(r.get("text", "")) for r in runs]
    if not any(is_punct):
        return
    for i, punct in enumerate(is_punct):
        if not punct:
            continue
        prev_size = None
        prev_dist = None
        for j in range(i - 1, -1, -1):
            if not is_punct[j] and runs[j].get("size") is not None:
                prev_size = int(runs[j]["size"])
                prev_dist = i - j
                break
        next_size = None
        next_dist = None
        for j in range(i + 1, len(runs)):
            if not is_punct[j] and runs[j].get("size") is not None:
                next_size = int(runs[j]["size"])
                next_dist = j - i
                break
        if prev_size is None and next_size is None:
            continue
        if prev_size is None:
            runs[i]["size"] = next_size
        elif next_size is None:
            runs[i]["size"] = prev_size
        elif prev_dist == next_dist:
            runs[i]["size"] = max(prev_size, next_size)
        elif prev_dist < next_dist:
            runs[i]["size"] = prev_size
        else:
            runs[i]["size"] = next_size


def _unify_run_sizes_by_color(runs: list[dict]) -> None:
    """Snap same-colour runs within a record to their dominant size.

    Mode C's per-word glyph_h measurement can vary 1-2 ladder steps
    between adjacent CJK characters of the same visual size. Within a
    single paragraph that's visible as ragged sizing, so nearby
    similar-size runs are collapsed onto one size.

    Rule: group runs by colour (exact match, after the upstream
    colour-cluster snap), within each group find the most common size,
    and snap any other run within ±2 ladder steps to it. Outliers (e.g.
    8 pt punctuation in a 16 pt body line) stay at their own size — the
    ladder-distance cap is the discriminator.
    """
    if not runs or len(runs) < 2:
        return
    by_color: dict[str, list[int]] = {}
    for i, r in enumerate(runs):
        if r.get("size") is None:
            continue
        by_color.setdefault(r["color"], []).append(i)
    for indices in by_color.values():
        if len(indices) < 2:
            continue
        sizes = [int(runs[i]["size"]) for i in indices]
        ladder_idx = {pt: i for i, pt in enumerate(_FONT_LADDER)}
        from collections import Counter
        counter = Counter(sizes)
        max_count = max(counter.values())
        candidates = [s for s, c in counter.items() if c == max_count]
        dominant = max(candidates)
        d_idx = ladder_idx.get(dominant, -1)
        if d_idx < 0:
            continue
        for i in indices:
            cur = int(runs[i]["size"])
            cur_idx = ladder_idx.get(cur, -1)
            if cur_idx < 0:
                continue
            if abs(cur_idx - d_idx) <= 2:
                runs[i]["size"] = dominant


def _glyph_height_in_bbox(orig_img: "np.ndarray", bg: "np.ndarray",
                          bbox: tuple[int, int, int, int]) -> int | None:
    """Measure vertical ink extent inside a text bbox.

    PaddleOCR's detection bbox often pads vertically for decorated stat
    numbers — drop shadows, glow effects and gradient borders inflate
    the box well past the actual glyph extent. Multiplying that padded
    height by α gives an inflated pt (e.g. `7分钟` bbox_h=97 →
    α=0.78 → 57 pt against a source rendered ~28-32 pt). This helper
    computes the real ink vertical range by thresholding the bbox
    region against the surrounding background colour.

    Returns the ink-row count (rows containing ≥ 2 stroke px), or None
    if sampling fails — caller falls back to raw bbox_h.
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = orig_img.shape[:2]
    x1 = max(0, min(w_img, int(x1))); x2 = max(0, min(w_img, int(x2)))
    y1 = max(0, min(h_img, int(y1))); y2 = max(0, min(h_img, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    region = orig_img[y1:y2, x1:x2]
    if region.size == 0:
        return None
    diff = np.abs(region.astype(int) - bg).max(axis=2)
    mask = diff > 35
    if int(mask.sum()) < 3:
        return None
    row_ink = mask.sum(axis=1) >= 2
    rows = np.where(row_ink)[0]
    if rows.size < 2:
        return None
    return int(rows.max() - rows.min() + 1)


def font_size_pt(text: str, bbox_w: int, bbox_h: int,
                 pt_per_px: float = 0.75) -> int:
    """Derive font size from bbox HEIGHT only (width ignored).

    Formula:  font_pt = bbox_h_px × pt_per_px × α
    where pt_per_px = slide_h_in × 72 / source_h_px and α reflects how
    tightly the OCR bbox hugs the glyph extent (≤ 4 chars tightest,
    longer text padded looser).

    A previous `is_stat` branch raised α to 0.96 for short digit+unit
    boxes like `63个`; it was calibrated against the old min(h, w)
    formula where width usually clipped the result. Under height-only
    that branch overshoots (40 pt for a glyph the source renders at
    ~36 pt). Length-bucket α only — single rule for all texts.

    Width is unreliable for mixed-glyph stats because PaddleOCR
    occasionally varies the per-glyph width estimate between visually
    identical lines (`524个` vs `63个` returned 48 vs 54 px wide).
    Height is the stable axis; cross-record drift gets cleaned up at
    PPTX-build time in apply_size_unification (ratio cap).
    """
    if len(text) <= 4:
        alpha = CAPTURE_FACTOR_SHORT
    elif len(text) <= 12:
        alpha = CAPTURE_FACTOR_MEDIUM
    else:
        alpha = CAPTURE_FACTOR_LONG
    raw = bbox_h * pt_per_px * alpha
    # Two-tier cap: most content text capped at 36 pt because OCR
    # detection bboxes pad heavily around decorated stat numbers
    # (shadows/glows lift bbox_h to 80-100 px → 56 pt raw). But the
    # cover/hero title can be genuinely huge, and capping it kills the
    # slide design. Use the raw value itself
    # as the discriminator: ≤ 60 pt is a suspect-decoration content
    # element → cap 36; > 60 pt is genuinely hero-scale → allow up to
    # the ladder max.
    if raw > 60.0:
        clamped = min(88.0, raw)
    else:
        clamped = min(36.0, raw)
    return _snap_to_ladder(max(8.0, clamped))


# =============================================================================
# Geometry + colour helpers
# =============================================================================


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _color_close(c1: str, c2: str, tol: int = 25) -> bool:
    """Compare two #RRGGBB strings with per-channel tolerance. Anti-aliasing
    causes slightly different sampled colors for two visually-identical
    text blocks; exact equality would prevent merging them."""
    if c1 == c2:
        return True
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return abs(r1 - r2) <= tol and abs(g1 - g2) <= tol and abs(b1 - b2) <= tol


def _is_stat_value(text: str) -> bool:
    """Stat-value heuristic: short text with digits and a unit suffix.
    Matches `1000亿元`, `24.22%`, `63个`, `29 PB`, etc."""
    compact = "".join(c for c in (text or "") if not c.isspace())
    if not any(c.isdigit() for c in compact) or len(compact) > 8:
        return False
    units = [
        "亿元", "PB", "MB", "GB", "TB",
        "分钟", "小时",
        "亿", "%", "个", "万", "轮", "条", "次", "倍", "年", "月", "天",
    ]
    for unit in units:
        if not compact.endswith(unit):
            continue
        number = compact[:-len(unit)] if unit else compact
        return bool(number) and all(c.isdigit() or c in ".,+-" for c in number)
    return False


def _should_apply_run_font_sizes(text: str) -> bool:
    """Only short numeric stats get per-run font sizes.

    Per-word glyph height is useful for `524个` or `7分钟`, where digits
    and unit suffixes are intentionally different sizes. It is harmful
    for headings and formulas (`Kona·核心洞察`, `O(N)→O(log N)`): tiny
    punctuation or Latin word boxes shrink individual runs and create
    visible drift. Those lines should keep one element-level size.
    """
    compact = "".join(c for c in (text or "") if not c.isspace())
    if not _is_stat_value(compact):
        return False
    formula_chars = set("()[]{}=<>→←+-*/")
    latin = sum(1 for c in compact if c.isascii() and c.isalpha())
    if any(c in formula_chars for c in compact) and latin >= 2:
        return False
    return True


def _text_width_units(text: str) -> float:
    units = 0.0
    for c in text or "":
        units += _char_width_unit(c)
    return units


def _char_width_unit(c: str) -> float:
    if c == "\n":
        return 0.0
    if c.isspace():
        return 0.35
    if c.isascii() and c.isalnum():
        return 0.56
    if c in ".,:;!|'`":
        return 0.25
    if c in "()[]{}（）":
        return 0.34
    if c in "-+/=":
        return 0.45
    if c in "→←•·‧":
        return 0.70
    return 1.0


def _split_box_by_text_units(
    text: str,
    box: list[int] | tuple[int, int, int, int],
) -> list[list[int]] | None:
    if not text:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    units = [_char_width_unit(c) for c in text]
    total = sum(units)
    if total <= 0:
        return None
    boxes: list[list[int]] = []
    cur = 0.0
    width = float(x2 - x1)
    for idx, (ch, unit) in enumerate(zip(text, units)):
        if unit <= 0:
            px1 = px2 = int(round(x1 + (cur / total) * width))
            boxes.append([px1, y1, px2, y2])
            continue
        start = cur
        cur += unit
        end = cur
        px1 = int(round(x1 + (start / total) * width))
        px2 = int(round(x1 + (end / total) * width))
        if idx == 0:
            px1 = x1
        if idx == len(text) - 1:
            px2 = x2
        if px2 <= px1:
            px2 = min(x2, px1 + 1)
        boxes.append([px1, y1, px2, y2])
    return boxes if len(boxes) == len(text) else None


def _char_boxes_from_words(
    text: str,
    words: list[str] | None,
    word_boxes: list[list[int]] | None,
) -> list[list[int]] | None:
    if (not text or words is None or word_boxes is None
            or len(words) != len(word_boxes)
            or "".join(words) != text):
        return None
    clipped: list[tuple[int, int, int, int]] = []
    for i, wb in enumerate(word_boxes):
        wx1, wy1, wx2, wy2 = (int(v) for v in wb)
        if i > 0:
            wx1 = max(wx1, int(word_boxes[i - 1][2]))
        if i + 1 < len(word_boxes):
            wx2 = min(wx2, int(word_boxes[i + 1][0]))
        if wx2 <= wx1:
            wx1, wy1, wx2, wy2 = (int(v) for v in wb)
        clipped.append((wx1, wy1, wx2, wy2))

    out: list[list[int]] = []
    for word, box in zip(words, clipped):
        if len(word) == 1:
            out.append([int(v) for v in box])
            continue
        split = _split_box_by_text_units(word, box)
        if split is None:
            return None
        out.extend(split)
    return out if len(out) == len(text) else None


def _derive_source_char_boxes(
    text: str,
    bbox: tuple[int, int, int, int],
    char_boxes: list[list[int]] | None,
    words: list[str] | None,
    word_boxes: list[list[int]] | None,
) -> list[list[int]] | None:
    if char_boxes is not None and len(char_boxes) == len(text):
        return [[int(v) for v in box] for box in char_boxes]
    from_words = _char_boxes_from_words(text, words, word_boxes)
    if from_words is not None:
        return from_words
    return _split_box_by_text_units(text, bbox)


def _estimated_text_width_px(text: str, size_pt: int, *,
                             pt_per_px: float, bold: bool) -> int:
    """Estimate the source-pixel width needed to avoid PPT text wrapping."""
    lines = str(text or "").split("\n")
    max_units = max((_text_width_units(line) for line in lines), default=0.0)
    if max_units <= 0 or pt_per_px <= 0:
        return 0
    weight = 1.08 if bold else 1.0
    # 1.10 absorbs LibreOffice/PowerPoint font metric differences without
    # making normal left-aligned OCR boxes visibly drift.
    return int(round((size_pt / pt_per_px) * max_units * 1.10 * weight))


def _estimated_runs_width_px(runs: list[dict], fallback_size_pt: int, *,
                             pt_per_px: float, bold: bool) -> int:
    """Estimate width for a line with run-level font sizes."""
    if not runs or pt_per_px <= 0:
        return 0
    weight = 1.08 if bold else 1.0
    line_width = 0.0
    max_width = 0.0
    for run in runs:
        text = str(run.get("text", ""))
        size = int(run.get("size") or fallback_size_pt)
        parts = text.split("\n")
        for idx, part in enumerate(parts):
            if idx > 0:
                max_width = max(max_width, line_width)
                line_width = 0.0
            line_width += (
                (size / pt_per_px)
                * _text_width_units(part)
                * 1.10
                * weight
            )
    max_width = max(max_width, line_width)
    return int(round(max_width))


def _glyph_height_in_char_box(source: np.ndarray,
                              char_box: list[int],
                              line_bg: np.ndarray) -> int | None:
    """Measure actual glyph ink height inside one OCR char/word box."""
    h_img, w_img = source.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in char_box)
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = source[y1:y2, x1:x2]
    if sub.size == 0:
        return None

    local_bg = line_bg
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] > 50
    if int(sat.sum()) > 0.45 * sat.size:
        local_bg = np.median(sub[sat], axis=0)
    diff = np.abs(sub.astype(int) - local_bg.astype(int)).max(axis=2)
    mask = diff > 35

    # White text on a saturated badge/header: the local bg is the fill,
    # and text is the low-saturation, higher-value foreground.
    if int(sat.sum()) > 0.45 * sat.size:
        bg_hsv = cv2.cvtColor(
            np.array([[local_bg]], dtype=np.uint8),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        light = (
            (hsv[:, :, 1] < 130)
            & (hsv[:, :, 2].astype(int) > int(bg_hsv[2]) + 25)
        )
        if int(light.sum()) >= 2:
            mask = light

    if int(mask.sum()) < 2:
        return None
    row_has_ink = mask.sum(axis=1) >= 1
    rows = np.where(row_has_ink)[0]
    if rows.size < 2:
        return None
    return int(rows.max() - rows.min() + 1)


def _char_colors_from_runs(text: str, runs: list[dict] | None,
                           fallback_color: str) -> list[str]:
    colors: list[str] = []
    if runs:
        for run in runs:
            color = run.get("color") or fallback_color
            colors.extend([color] * len(str(run.get("text", ""))))
    if len(colors) != len(text):
        colors = [fallback_color] * len(text)
    # If per-character sampling produced runs but they all have the same
    # sampled colour, there is no real inline colour variation. Use the
    # line-level classified colour instead; it snaps anti-aliased dark
    # blue/gray samples back to the canonical PPT palette.
    elif len({c for c in colors if c}) <= 1:
        colors = [fallback_color] * len(text)
    return colors


def _build_runs_from_char_sizes(text: str, colors: list[str],
                                sizes: list[int],
                                bolds: list[bool] | None = None) -> list[dict]:
    runs: list[dict] = []
    cur_text = ""
    cur_color: str | None = None
    cur_size: int | None = None
    cur_bold: bool | None = None
    if bolds is None or len(bolds) != len(text):
        bolds = [False] * len(text)
    for c, color, size, is_bold in zip(text, colors, sizes, bolds):
        if (cur_text and color == cur_color and size == cur_size
                and bool(is_bold) == bool(cur_bold)):
            cur_text += c
            continue
        if cur_text:
            run = {"text": cur_text,
                   "color": cur_color,
                   "size": int(cur_size)}
            if cur_bold:
                run["bold"] = True
            runs.append(run)
        cur_text = c
        cur_color = color
        cur_size = int(size)
        cur_bold = bool(is_bold)
    if cur_text:
        run = {"text": cur_text,
               "color": cur_color,
               "size": int(cur_size)}
        if cur_bold:
            run["bold"] = True
        runs.append(run)
    return runs


def mixed_size_runs_from_char_boxes(
    text: str,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    char_boxes: list[list[int]] | None,
    existing_runs: list[dict] | None,
    fallback_color: str,
    *,
    bold: bool,
    pt_per_px: float,
) -> dict | None:
    """Detect common same-line mixed-size patterns and emit run sizes.

    Many generated PPT screenshots use a larger leading label followed by
    a smaller parenthetical explanation in the same OCR line. A single
    line-level render fit picks a compromise size. Char-level ink heights
    let us split the line into PPT runs with different sizes.
    """
    if not char_boxes or len(char_boxes) != len(text) or pt_per_px <= 0:
        return None
    open_indices = [i for i, c in enumerate(text) if c in "（("]
    if not open_indices:
        return None
    open_idx = open_indices[0]
    if open_idx < 2 or len(text) - open_idx < 3:
        return None

    line_bg = _sample_background_for_bbox(source, bbox)
    raw_heights: list[int | None] = [
        _glyph_height_in_char_box(source, cb, line_bg)
        for cb in char_boxes
    ]

    def visible_heights(indices) -> list[int]:
        vals: list[int] = []
        for i in indices:
            c = text[i]
            h = raw_heights[i]
            if h is None or not c.strip() or _is_punct_run(c):
                continue
            vals.append(int(h))
        return vals

    prefix = visible_heights(range(0, open_idx))
    suffix = visible_heights(range(open_idx + 1, len(text)))
    if len(prefix) < 2 or len(suffix) < 2:
        return None
    prefix_h = float(np.median(prefix))
    suffix_h = float(np.median(suffix))
    if prefix_h <= 0 or suffix_h <= 0:
        return None
    # Require a visible size step. This avoids splitting lines where glyph
    # shape alone makes some characters 1-2 px shorter.
    if suffix_h > prefix_h * 0.90 or prefix_h - suffix_h < 3:
        return None

    def _union_box(boxes: list[list[int]]) -> tuple[int, int, int, int]:
        return (
            min(int(b[0]) for b in boxes),
            min(int(b[1]) for b in boxes),
            max(int(b[2]) for b in boxes),
            max(int(b[3]) for b in boxes),
        )

    prefix_text = text[:open_idx]
    suffix_text = text[open_idx:]
    prefix_box = _union_box(char_boxes[:open_idx])
    suffix_box = _union_box(char_boxes[open_idx:])
    prefix_init = font_size_pt(prefix_text,
                               prefix_box[2] - prefix_box[0],
                               prefix_box[3] - prefix_box[1],
                               pt_per_px=pt_per_px)
    suffix_init = font_size_pt(suffix_text,
                               suffix_box[2] - suffix_box[0],
                               suffix_box[3] - suffix_box[1],
                               pt_per_px=pt_per_px)
    prefix_fit = fit_text_render(
        prefix_text, source, prefix_box,
        initial_size=prefix_init,
        initial_bold=bold,
        initial_font=default_ppt_font(),
        pt_per_px=pt_per_px,
    )
    suffix_fit = fit_text_render(
        suffix_text, source, suffix_box,
        initial_size=suffix_init,
        initial_bold=bold,
        initial_font=default_ppt_font(),
        pt_per_px=pt_per_px,
    )

    base_size = int(prefix_fit["size"]) if prefix_fit else int(prefix_init)
    suffix_size = int(suffix_fit["size"]) if suffix_fit else int(suffix_init)
    if base_size - suffix_size < 2:
        return None

    def _mostly_cjk(value: str) -> bool:
        chars = [c for c in value if c.strip() and not _is_punct_run(c)]
        if not chars:
            return False
        cjk = sum(1 for c in chars if 0x4E00 <= ord(c) <= 0x9FFF)
        return cjk >= max(2, int(math.ceil(len(chars) * 0.70)))

    def _ink_density(box: tuple[int, int, int, int],
                     glyph_h: float) -> float | None:
        if glyph_h <= 0:
            return None
        metrics = _target_text_metrics(source, box)
        if metrics is None or metrics["ink_w"] <= 0:
            return None
        return float(metrics["ink_area"]) / max(
            1.0, float(metrics["ink_w"]) * float(glyph_h))

    prefix_bold = bool(bold)
    suffix_bold = bool(bold)
    prefix_density = _ink_density(prefix_box, prefix_h)
    if (not prefix_bold
            and _mostly_cjk(prefix_text)
            and prefix_h >= 10
            and prefix_density is not None
            and prefix_density >= 0.52):
        prefix_bold = True

    sizes = [base_size if i < open_idx else suffix_size
             for i in range(len(text))]
    bolds = [prefix_bold if i < open_idx else suffix_bold
             for i in range(len(text))]
    colors = _char_colors_from_runs(text, existing_runs, fallback_color)
    return {
        "runs": _build_runs_from_char_sizes(text, colors, sizes, bolds),
        "base_size": base_size,
        "suffix_size": suffix_size,
        "prefix_h": round(prefix_h, 2),
        "suffix_h": round(suffix_h, 2),
        "prefix_bold": bool(prefix_bold),
        "suffix_bold": bool(suffix_bold),
        "prefix_density": (
            round(float(prefix_density), 4)
            if prefix_density is not None else None
        ),
        "prefix_fit_score": (
            round(float(prefix_fit["score"]), 4) if prefix_fit else None
        ),
        "suffix_fit_score": (
            round(float(suffix_fit["score"]), 4) if suffix_fit else None
        ),
    }


def _font_candidates() -> list[dict]:
    """Local render fonts that also have usable PPT font-family names."""
    global _FONT_CANDIDATES_CACHE
    if _FONT_CANDIDATES_CACHE is not None:
        return _FONT_CANDIDATES_CACHE

    yahei_path = _fontconfig_font_path("Microsoft YaHei:style=Regular")
    # Keep the default first. Non-default faces are allowed to win only
    # when their rendered pixels materially match the source better. Use
    # one stable CJK sans candidate to avoid line-by-line face flipping
    # inside the same paragraph.
    specs: list[tuple[str, str, float]] = []
    if yahei_path:
        specs.append(("Microsoft YaHei", yahei_path, 0.00))
    specs.extend([
        ("Arial Unicode MS", "/Library/Fonts/Arial Unicode.ttf", 0.00),
        ("Arial Unicode MS",
         "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0.02),
        ("Hiragino Sans GB", "/System/Library/Fonts/Hiragino Sans GB.ttc",
         0.06),
    ])
    if not Path("/System/Library/Fonts/Hiragino Sans GB.ttc").exists():
        specs.append(("Heiti TC", "/System/Library/Fonts/STHeiti Medium.ttc",
                      0.06))
    seen_paths: set[str] = set()
    out: list[dict] = []
    for ppt_name, raw_path, penalty in specs:
        path = Path(raw_path)
        if not path.exists() or str(path) in seen_paths:
            continue
        try:
            ImageFont.truetype(str(path), 16)
        except OSError:
            continue
        seen_paths.add(str(path))
        out.append({
            "ppt_name": ppt_name,
            "path": str(path),
            "penalty": float(penalty),
        })
    _FONT_CANDIDATES_CACHE = out
    return out


def _load_font(path: str, pixel_size: int) -> ImageFont.FreeTypeFont | None:
    pixel_size = max(1, int(pixel_size))
    key = (path, pixel_size)
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(path, pixel_size)
        except OSError:
            return None
    return _FONT_CACHE[key]


def _render_text_metrics(text: str, font_path: str, size_pt: int,
                         pt_per_px: float, bold: bool) -> dict | None:
    """Render one text line locally and return its ink bbox in source px.

    The layout stores font size in points, but the source image is in
    pixels. ``pt_per_px`` converts PPT points back to source pixels so the
    rendered mask can be compared directly with the OCR crop.
    """
    if not text or "\n" in text or pt_per_px <= 0:
        return None
    pixel_size = max(1, int(round(float(size_pt) / pt_per_px)))
    cache_key = (text, font_path, pixel_size, bool(bold))
    if cache_key in _TEXT_RENDER_CACHE:
        return _TEXT_RENDER_CACHE[cache_key]

    font = _load_font(font_path, pixel_size)
    if font is None:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None

    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except UnicodeEncodeError:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    if not bbox:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None

    bx1, by1, bx2, by2 = (int(v) for v in bbox)
    pad = max(4, pixel_size // 3)
    canvas_w = max(2, bx2 - bx1 + pad * 2 + (1 if bold else 0))
    canvas_h = max(2, by2 - by1 + pad * 2)
    origin = (pad - bx1, pad - by1)
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    draw.text(origin, text, font=font, fill=255)
    if bold:
        # Synthetic one-pixel emboldening. It approximates how PPT/LO
        # fattens regular CJK fonts when only a regular face is available.
        draw.text((origin[0] + 1, origin[1]), text, font=font, fill=255)

    arr = np.array(img)
    ys, xs = np.where(arr > 16)
    if ys.size == 0 or xs.size == 0:
        _TEXT_RENDER_CACHE[cache_key] = None
        return None
    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1
    rel_bbox = [
        int(x_min - origin[0]),
        int(y_min - origin[1]),
        int(x_max - origin[0]),
        int(y_max - origin[1]),
    ]
    mask_crop = arr[y_min:y_max, x_min:x_max] > 16
    metrics = {
        "pixel_size": pixel_size,
        "ink_bbox": rel_bbox,
        "ink_w": int(x_max - x_min),
        "ink_h": int(y_max - y_min),
        "ink_area": int(mask_crop.sum()),
        "mask": mask_crop,
    }
    _TEXT_RENDER_CACHE[cache_key] = metrics
    return metrics


def _sample_background_for_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = img.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    gap = 2
    ring = 6
    inner_top = max(0, y1 - gap)
    inner_bot = min(h_img, y2 + gap)
    inner_left = max(0, x1 - gap)
    inner_right = min(w_img, x2 + gap)
    samples = []
    if inner_top - ring >= 0:
        samples.append(img[inner_top - ring:inner_top, inner_left:inner_right])
    if inner_bot + ring <= h_img:
        samples.append(img[inner_bot:inner_bot + ring, inner_left:inner_right])
    if inner_left - ring >= 0:
        samples.append(img[inner_top:inner_bot, inner_left - ring:inner_left])
    if inner_right + ring <= w_img:
        samples.append(img[inner_top:inner_bot, inner_right:inner_right + ring])
    if samples:
        pixels = np.concatenate([s.reshape(-1, 3) for s in samples if s.size])
        if len(pixels):
            from collections import Counter
            quant = (pixels // 16) * 16
            mode_q = np.array(Counter(map(tuple, quant)).most_common(1)[0][0])
            diff = np.abs(pixels.astype(int) - mode_q).max(axis=1)
            close = pixels[diff <= 30]
            if len(close) >= max(8, int(len(pixels) * 0.2)):
                return np.median(close, axis=0)
            return np.median(pixels, axis=0)
    if x2 > x1 and y2 > y1:
        region = img[y1:y2, x1:x2]
        if region.size:
            border = np.concatenate([
                region[:1, :].reshape(-1, 3),
                region[-1:, :].reshape(-1, 3),
                region[:, :1].reshape(-1, 3),
                region[:, -1:].reshape(-1, 3),
            ])
            if len(border):
                return np.median(border, axis=0)
    return np.array([255.0, 255.0, 255.0])


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
    px1 = max(0, x1 - pad)
    px2 = min(w_img, x2 + pad)
    py1 = max(0, y1 - pad)
    py2 = min(h_img, y2 + pad)
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
        if int(np.max(np.abs(container_bg.astype(int) - bg.astype(int)))) > 40:
            bg = container_bg
            component = largest_component if largest_component is not None else sat_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                component * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
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

    ix1 = int(cols.min())
    ix2 = int(cols.max()) + 1
    iy1 = int(rows.min())
    iy2 = int(rows.max()) + 1
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
        render_mask.astype(np.uint8) * 255,
        (tw, th),
        interpolation=cv2.INTER_AREA,
    ) > 64
    target = target_mask.astype(bool)
    inter = int((target & resized).sum())
    union = int((target | resized).sum())
    if union <= 0:
        return 0.5
    return 1.0 - (inter / union)


def _binary_size_candidates(
    text: str,
    font_path: str,
    bold: bool,
    target_h: int,
    initial_size: int,
    pt_per_px: float,
) -> list[int]:
    """Fast-lock likely pt sizes by binary-searching rendered ink height."""
    lo, hi = 5, 96
    for _ in range(8):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        metrics = _render_text_metrics(text, font_path, mid, pt_per_px, bold)
        mid_h = int(metrics["ink_h"]) if metrics else 0
        if mid_h < target_h:
            lo = mid + 1
        else:
            hi = mid
    center = lo
    candidates = set(range(center - 3, center + 4))
    candidates.update(range(int(initial_size) - 4, int(initial_size) + 5))
    candidates.update(v for v in _FONT_LADDER
                      if abs(v - int(initial_size)) <= 8)
    return sorted(v for v in candidates if 5 <= v <= 96)


def fit_text_render(
    text: str,
    source: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    initial_size: int,
    initial_bold: bool,
    initial_font: str,
    pt_per_px: float,
) -> dict | None:
    """Choose font/size/position by comparing render masks to source ink."""
    if not text or "\n" in text:
        return None
    target = _target_text_metrics(source, bbox)
    if target is None or target["ink_h"] < 3 or target["ink_w"] < 3:
        return None
    fonts = _font_candidates()
    if not fonts:
        return None
    if _is_punct_run(text):
        fonts = [f for f in fonts if f["ppt_name"] == initial_font] or fonts[:1]

    best: dict | None = None
    bold_options = [bool(initial_bold)]
    if text.strip() and len(text.strip()) <= 40:
        bold_options.append(not bool(initial_bold))

    for font_idx, font in enumerate(fonts):
        font_path = font["path"]
        font_penalty = float(font.get("penalty", 0.0))
        if font["ppt_name"] != initial_font:
            font_penalty += 0.02 + font_idx * 0.005
        for bold in bold_options:
            size_candidates = _binary_size_candidates(
                text, font_path, bold, int(target["ink_h"]),
                int(initial_size), pt_per_px,
            )
            for size in size_candidates:
                metrics = _render_text_metrics(text, font_path, size,
                                               pt_per_px, bold)
                if not metrics:
                    continue
                h_err = abs(metrics["ink_h"] - target["ink_h"]) / max(
                    1, target["ink_h"])
                w_err = abs(metrics["ink_w"] - target["ink_w"]) / max(
                    1, target["ink_w"])
                area_err = abs(metrics["ink_area"] - target["ink_area"]) / max(
                    1, target["ink_area"])
                shape_err = _mask_shape_error(target["mask"], metrics["mask"])
                size_penalty = abs(size - int(initial_size)) / max(
                    16.0, float(initial_size))
                bold_penalty = 0.0 if bold == bool(initial_bold) else 0.06
                score = (
                    0.40 * h_err
                    + 0.30 * w_err
                    + 0.12 * min(1.5, area_err)
                    + 0.12 * shape_err
                    + 0.035 * min(2.0, size_penalty)
                    + font_penalty
                    + bold_penalty
                )
                candidate = {
                    "score": float(score),
                    "height_err": float(h_err),
                    "width_err": float(w_err),
                    "shape_err": float(shape_err),
                    "font": font["ppt_name"],
                    "font_path": font_path,
                    "size": int(size),
                    "bold": bool(bold),
                    "metrics": metrics,
                    "target": target,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None:
        return None
    # If the best render is still far from the source ink, keep the
    # conservative bbox-height fallback rather than trusting a bad match.
    if (best["score"] > 0.55
            and (best["height_err"] > 0.35 or best["width_err"] > 0.55)):
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    metrics = best["metrics"]
    target = best["target"]
    rx1, ry1, rx2, ry2 = (int(v) for v in metrics["ink_bbox"])
    tx1, ty1, tx2, ty2 = (int(v) for v in target["ink_bbox"])

    raw_x = tx1 - rx1
    raw_y = ty1 - ry1
    max_dx = max(5, int(round(0.35 * bbox_h)))
    max_dy = max(6, int(round(0.75 * bbox_h)))
    fit_x = int(round(max(x1 - max_dx, min(x1 + max_dx, raw_x))))
    fit_y = int(round(max(y1 - max_dy, min(y1 + max_dy, raw_y))))
    fit_x = max(0, min(source.shape[1] - 1, fit_x))
    fit_y = max(0, min(source.shape[0] - 1, fit_y))

    render_w = int(metrics["ink_w"])
    line_px = float(best["size"]) / max(0.01, pt_per_px)
    text_w = max(
        bbox_w + 6,
        render_w + 8,
        _estimated_text_width_px(
            text, int(best["size"]),
            pt_per_px=pt_per_px,
            bold=bool(best["bold"]),
        ) + 8,
    )
    text_h = max(
        bbox_h + 4,
        int(round(max(float(ry2) + 4.0, line_px * 1.35))),
    )

    return {
        "font": best["font"],
        "size": int(best["size"]),
        "bold": bool(best["bold"]),
        "box": [fit_x, fit_y, int(text_w), int(text_h)],
        "score": round(float(best["score"]), 4),
        "target_ink": [int(tx1), int(ty1), int(tx2), int(ty2)],
        "render_ink": [int(rx1), int(ry1), int(rx2), int(ry2)],
    }


_ICON_PAD_ROLES = {"small_icon", "preserve_visual_icon", "subicon", "line_subicon"}
_SPARSE_VISUAL_ROLES = _ICON_PAD_ROLES | {"thin_rule", "outline"}


def _intersection_area(a: tuple[int, int, int, int],
                       b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ox1, oy1 = max(ax1, bx1), max(ay1, by1)
    ox2, oy2 = min(ax2, bx2), min(ay2, by2)
    if ox2 <= ox1 or oy2 <= oy1:
        return 0
    return (ox2 - ox1) * (oy2 - oy1)


def _overlaps_text(box: tuple[int, int, int, int],
                   text_boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(_intersection_area(box, tb) > 0 for tb in text_boxes)


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


def _bgr_to_hex(color: np.ndarray | list | tuple) -> str:
    b, g, r = (int(v) for v in color[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _line_art_alpha(crop_bgr: np.ndarray) -> np.ndarray:
    """Alpha mask for sparse line icons without keeping rectangular bg.

    The foreground can be a pure stroke icon, a white badge on a coloured
    title bar, or a filled warning triangle with white holes. We preserve
    filled foreground islands and their enclosed holes, but leave ordinary
    surrounding panel/background pixels transparent.
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
    bg = np.median(border, axis=0) if len(border) else np.array([255, 255, 255])
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
    # If the mask sampled mostly white interior pixels, fall back to the
    # deck's common pale blue frame colour instead of emitting invisible
    # white lines.
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


def _outline_should_be_native_shape(bbox: tuple[int, int, int, int]) -> bool:
    """Use native round-rect only for card-like outlines.

    Near-square outline masks are often circular icon containers or central
    rings. Rendering those as a PowerPoint rounded rectangle creates an
    extra visible box, so they should stay as alpha-masked PNGs that
    preserve the original contour.
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
        elif start is not None and gap_start is not None and idx - gap_start >= min_gap:
            if gap_start - start >= 28:
                ranges.append((start, gap_start))
            start = None
            gap_start = None
    if start is not None:
        end = gap_start if gap_start is not None and h - gap_start >= min_gap else h
        if end - start >= 28:
            ranges.append((start, end))

    if len(ranges) <= 1:
        return [(x1, y1, x2, y2)]
    # Keep only real card-height rows; tiny rule/text fragments are not
    # independent panels.
    boxes = [(x1, y1 + rs, x2, y1 + re) for rs, re in ranges if re - rs >= 36]
    if len(boxes) <= 1:
        return [(x1, y1, x2, y2)]
    return boxes


# =============================================================================
# Asset bbox shaping: icon padding + short-stat trim
# =============================================================================


def _pad_icon_bbox(
    bbox: tuple[int, int, int, int],
    probe_img: np.ndarray,
    text_boxes: list[tuple[int, int, int, int]],
    role: str | None,
    *,
    allow_text_overlap: bool = False,
) -> tuple[int, int, int, int]:
    """Loosen icon-ish crops when the surrounding pixels are background.

    Tight detection boxes often clip antialiasing or make the resulting PPT
    image object awkwardly precise. Expanding only through plain background
    avoids baking nearby editable text into the asset. When the crop source
    is already text-erased, OCR text boxes can be ignored because the text
    shapes render above every image later.
    """
    h_img, w_img = probe_img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(w_img, x1))
    x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1))
    y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x2, y2)

    w, h = x2 - x1, y2 - y1
    explicit_icon = role in _ICON_PAD_ROLES
    implicit_cv2_icon = role in {None, ""}
    if not explicit_icon and not implicit_cv2_icon:
        return (x1, y1, x2, y2)

    # Keep this scoped to icon-ish elements. Since the default cv2 path
    # crops from text-erased images, medium icons can safely receive a
    # little more background. Very large regions are probably cards,
    # screenshots, or whole compositions rather than standalone icons.
    max_side = 340 if allow_text_overlap else 280
    max_area = 80000 if allow_text_overlap else 52000
    if w > max_side or h > max_side or w * h > max_area:
        return (x1, y1, x2, y2)
    ratio = 0.06 if max(w, h) > 140 else 0.08
    pad_max = 10 if allow_text_overlap else 8
    pad = max(2, min(pad_max, int(round(max(w, h) * ratio))))
    if role == "subicon":
        pad = min(pad, 6)

    nx1, ny1, nx2, ny2 = x1, y1, x2, y2
    if x1 > 0:
        px1 = max(0, x1 - pad)
        side = (px1, y1, x1, y2)
        cand = (px1, y1, x2, y2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            nx1 = px1
    if x2 < w_img:
        px2 = min(w_img, x2 + pad)
        side = (x2, y1, px2, y2)
        cand = (nx1, y1, px2, y2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            nx2 = px2
    if y1 > 0:
        py1 = max(0, y1 - pad)
        side = (nx1, py1, nx2, y1)
        cand = (nx1, py1, nx2, y2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            ny1 = py1
    if y2 < h_img:
        py2 = min(h_img, y2 + pad)
        side = (nx1, y2, nx2, py2)
        cand = (nx1, ny1, nx2, py2)
        if (
            (allow_text_overlap or not _overlaps_text(cand, text_boxes))
            and _simple_background_strip(probe_img, side)
        ):
            ny2 = py2

    out = (nx1, ny1, nx2, ny2)
    if not allow_text_overlap and _overlaps_text(out, text_boxes):
        return (x1, y1, x2, y2)
    return out


def _trim_short_stat_text_bbox(
    text: str,
    bbox: tuple[int, int, int, int],
    source: np.ndarray,
) -> tuple[int, int, int, int]:
    """Shrink OCR boxes that include a leading icon before a short stat.

    OCR often reports `7分钟` / `16小时` together with the clock icon on the
    left. The eraser now preserves that icon; the editable text box must
    start after it or the rendered text will sit on top of the icon.
    """
    if len(text) > 5 or not any(c.isdigit() for c in text):
        return bbox
    x1, y1, x2, y2 = bbox
    h_img, w_img = source.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return bbox
    region = source[y1:y2, x1:x2]
    h, w = region.shape[:2]
    # All pixel thresholds in this function are calibrated against a
    # 720-tall reference image. `scale` rescales them so the function
    # works equivalently at 1080-tall, 600-tall, or any other input
    # resolution.
    scale = h_img / 720.0
    if h < 20 * scale or w < 40 * scale:
        return bbox
    border = np.concatenate([
        region[:3, :].reshape(-1, 3),
        region[-3:, :].reshape(-1, 3),
        region[:, :3].reshape(-1, 3),
        region[:, -3:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    diff = np.abs(region.astype(int) - bg.astype(int)).max(axis=2)
    fg = cv2.morphologyEx((diff > 30).astype(np.uint8) * 255,
                          cv2.MORPH_CLOSE,
                          np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    # Per-column foreground presence is needed for the gap check below.
    col_fg = fg.sum(axis=0)
    # Convert reference-720 pixel thresholds to this image's scale.
    min_area = 80 * scale * scale
    min_dim = max(3, int(round(3 * scale)))
    max_icon_dim = 75 * scale
    gap_window = max(12, int(round(12 * scale)))
    min_gap_run = max(3, int(round(3 * scale)))
    trim_to = x1
    for i in range(1, n):
        lx, ly, lw, lh, area = (int(v) for v in stats[i])
        if area < min_area or lw <= min_dim or lh <= min_dim:
            continue
        aspect = lw / max(1, lh)
        touches_left_icon_zone = lx <= max(4 * scale, int(0.12 * w))
        # Real glyph-prefix icons (clock, location pin, $, ¥) are
        # close to square — aspect ≥ 0.60. Leading digits like the
        # `1` in `1亿人` (aspect ~0.38) are tall and narrow; the
        # earlier 0.28 lower bound let them through and we'd trim
        # the digit off the stat.
        iconish = (
            touches_left_icon_zone
            and lh >= 0.35 * h
            and max(lw, lh) <= max_icon_dim
            and 0.60 <= aspect <= 1.60
        )
        if not iconish:
            continue
        # Even when the leading CC looks square-ish, require a clear
        # horizontal gap (≥ ~3 px @ 720-ref) right after it. Genuine
        # icon-then-text has that gap; a wide bold digit that
        # accidentally hits the aspect window would have its next
        # glyph immediately adjacent.
        gap_start = lx + lw
        gap_end = min(w, gap_start + gap_window)
        if gap_end <= gap_start:
            continue
        run = 0
        ok = False
        for c in range(gap_start, gap_end):
            if col_fg[c] == 0:
                run += 1
                if run >= min_gap_run:
                    ok = True
                    break
            else:
                run = 0
        if not ok:
            continue
        trim_to = max(trim_to, x1 + lx + lw + int(round(4 * scale)))
    if trim_to <= x1 + 4 * scale or trim_to >= x2 - 12 * scale:
        return bbox
    return (trim_to, y1, x2, y2)


# =============================================================================
# Text post-processing: unify similar sizes + merge multi-line groups
# =============================================================================


def unify_group_sizes(text_records: list[dict]) -> list[dict]:
    """Bold-only group unification. Font sizes are NEVER overridden here.

    PaddleOCR's bbox is the authoritative source for both position and
    font size — per-bbox `font_size_pt(text, bbox_w, bbox_h)` already
    derives size directly from bbox geometry. This pass used to also
    snap each record to its group's median size to smooth out 1-2 px
    bbox-height jitter between visually-similar texts, but that risks
    washing out real per-text differences the bbox correctly captured.

    What remains here:
      - Group records by category + fuzzy-color + similar bbox height
        (same buckets as before)
      - For each group with ≥ 2 members, take a majority vote on the
        bold flag and apply it to every member. Bold is derived from
        stroke-pixel density, which is noisy per-bbox (thin glyphs like
        `%` always under-vote bold), so this majority vote is a legit
        denoising step that doesn't override geometric bbox signal.

    Categories:
      - `stat`: numeric values with units (`1000亿元`, `24.22%`)
      - `short`, `medium`, `long`: by char count (mirrors font_size_pt)

    Color comparison is fuzzy (per-channel ±25) so anti-aliasing variants
    (e.g. `#D43E3E` vs `#D63D3D` red) end up in the same group.
    """
    if not text_records:
        return text_records

    def category(text: str) -> str:
        if _is_stat_value(text):
            return "stat"
        n = len(text)
        if n <= 4:
            return "short"
        if n <= 12:
            return "medium"
        return "long"

    # Build groups by (category, fuzzy-color, similar bbox height) — bbox
    # height similarity is critical: a cover title (bbox h≈150) must NOT
    # cluster with same-color info-row text (bbox h≈30). Without the height
    # gate, the median would pull the title down to 16 pt. Bold flag is
    # intentionally NOT a grouping key (density-based detection flips for
    # thin glyphs like `%`).
    groups: list[list[int]] = []
    assigned = [False] * len(text_records)
    for i, r in enumerate(text_records):
        if assigned[i]:
            continue
        cat_i = category(r["text"])
        col_i = r["color"]
        h_i = r["box"][3]
        bucket = [i]
        assigned[i] = True
        for j, other in enumerate(text_records):
            if assigned[j]:
                continue
            if category(other["text"]) != cat_i:
                continue
            if not _color_close(other["color"], col_i):
                continue
            # bbox height must be within ±25 % of the seed
            h_j = other["box"][3]
            if max(h_i, h_j) > min(h_i, h_j) * 1.25:
                continue
            bucket.append(j)
            assigned[j] = True
        groups.append(bucket)

    # Bold-only majority vote per group. Size is left untouched —
    # font_size_pt(bbox_w, bbox_h) already derived it from the
    # authoritative PaddleOCR bbox.
    out = [dict(r) for r in text_records]
    for bucket in groups:
        if len(bucket) < 2:
            continue
        bold_votes = sum(1 for i in bucket if out[i]["bold"])
        unified_bold = bool(bold_votes * 2 >= len(bucket))
        for i in bucket:
            out[i]["bold"] = unified_bold
    return out


def merge_multiline_texts(text_records: list[dict]) -> list[dict]:
    """Merge stacked text records that share style and column position into
    a single multi-line text element (newline-joined).

    A pair (a, b) where a is above b qualifies for merge when:
      - x-overlap ≥ 20 % of the narrower box (same column),
      - vertical gap < 0.8 × the larger line height (one line away or closer),
      - colors close per channel within ±25 (handles anti-aliasing variance),
      - font sizes within ±2 pt after snap-to-ladder,
      - same bold flag.

    After merging, all records in the group are unified to the most common
    (size, color) so a paragraph reads with consistent style. Returns a new
    list of records.
    """
    items = sorted(text_records, key=lambda r: (r["box"][1], r["box"][0]))
    used = [False] * len(items)
    out: list[dict] = []
    for i, base in enumerate(items):
        if used[i]:
            continue
        used[i] = True
        bx, by, bw, bh = base["box"]
        group = [base]
        gx1, gy1, gx2, gy2 = bx, by, bx + bw, by + bh
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(items):
                if used[j]:
                    continue
                ox, oy, ow, oh = other["box"]
                ox2, oy2 = ox + ow, oy + oh
                if other["bold"] != base["bold"]:
                    continue
                if abs(other["size"] - base["size"]) > 2:
                    continue
                if not _color_close(other["color"], base["color"]):
                    continue
                # X overlap ≥ 20 % of narrower column
                x_overlap = min(gx2, ox2) - max(gx1, ox)
                if x_overlap < min(gx2 - gx1, ow) * 0.20:
                    continue
                # Vertical gap to current group bbox < 0.8 × larger line height
                line_h = max(gy2 - gy1, oh)
                gap = oy - gy2 if oy >= gy2 else gy1 - oy2
                if gap > line_h * 0.8:
                    continue
                # Accept merge
                group.append(other)
                used[j] = True
                gx1, gy1 = min(gx1, ox), min(gy1, oy)
                gx2, gy2 = max(gx2, ox2), max(gy2, oy2)
                changed = True
        if len(group) == 1:
            out.append(base)
        else:
            group.sort(key=lambda g: g["box"][1])
            # Unify style: pick most common size and (channel-rounded) color
            from collections import Counter
            size_counts = Counter(g["size"] for g in group)
            common_size = size_counts.most_common(1)[0][0]
            color_counts = Counter(g["color"] for g in group)
            common_color = color_counts.most_common(1)[0][0]
            merged = dict(base)
            merged["text"] = "\n".join(g["text"] for g in group)
            merged["box"] = [int(gx1), int(gy1), int(gx2 - gx1), int(gy2 - gy1)]
            merged["size"] = int(common_size)
            merged["color"] = common_color
            merged["line_spacing"] = 1.15
            out.append(merged)
    return out


_BULLET_PREFIX_CHARS = {"·", "•", "‧", "∙", "●", "○", "◦", "▪", "▫"}
_ORDERED_PREFIX_RE = re.compile(r"^\s*(\(?)(\d{1,3})([.)、．])\)?\s*")


def _list_prefix(text: str) -> dict | None:
    if not text:
        return None
    stripped_left = text.lstrip()
    leading_ws = len(text) - len(stripped_left)
    if stripped_left[:1] in _BULLET_PREFIX_CHARS:
        return {
            "kind": "bullet",
            "marker_chars": leading_ws + 1,
            "marker_text": stripped_left[:1],
            "marker": "•",
        }
    match = _ORDERED_PREFIX_RE.match(text)
    if match:
        suffix = match.group(3)
        if suffix in {".", "．"} and match.end() < len(text) and text[match.end()].isdigit():
            return None
        auto_type = "arabicPeriod"
        if suffix == ")":
            auto_type = "arabicParenR"
        return {
            "kind": "ordered",
            "marker_chars": match.end(),
            "marker_text": text[:match.end()],
            "start": int(match.group(2)),
            "auto_type": auto_type,
        }
    return None


def _list_marker_geometry(el: dict, marker_chars: int) -> tuple[float, float] | None:
    chars = el.get("source_chars")
    boxes = el.get("source_char_boxes")
    if chars and boxes and len(chars) == len(boxes):
        marker_indices = [
            i for i in range(min(marker_chars, len(chars)))
            if str(chars[i]).strip()
        ]
        body_indices = [
            i for i in range(marker_chars, len(chars))
            if str(chars[i]).strip()
        ]
        if marker_indices and body_indices:
            marker_x = min(float(boxes[i][0]) for i in marker_indices)
            body_x = min(float(boxes[i][0]) for i in body_indices)
            if body_x > marker_x:
                return marker_x, body_x
    bbox = el.get("source_bbox")
    if bbox and len(bbox) == 4:
        x1, y1, _x2, y2 = (float(v) for v in bbox)
        h = max(1.0, y2 - y1)
        return x1 + max(1.0, h * 0.22), x1 + max(12.0, h * 1.15)
    box = el.get("box")
    if box and len(box) == 4:
        x, _y, _w, h = (float(v) for v in box)
        return x + max(1.0, h * 0.18), x + max(12.0, h * 1.05)
    return None


def _same_list_column(a: dict, b: dict) -> bool:
    ax, ay, aw, ah = (float(v) for v in a.get("box", [0, 0, 0, 0]))
    bx, by, bw, bh = (float(v) for v in b.get("box", [0, 0, 0, 0]))
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    narrower = max(1.0, min(aw, bw))
    x_close = abs(ax - bx) <= max(18.0, min(40.0, max(ah, bh) * 2.0))
    return overlap / narrower >= 0.35 or x_close


def _has_list_neighbour(idx: int, candidates: list[tuple[int, dict, dict]]) -> bool:
    _orig_idx, el, prefix = candidates[idx]
    x, y, w, h = (float(v) for v in el.get("box", [0, 0, 0, 0]))
    for j, (_other_idx, other, other_prefix) in enumerate(candidates):
        if j == idx:
            continue
        if other_prefix.get("kind") != prefix.get("kind"):
            continue
        if not _same_list_column(el, other):
            continue
        ox, oy, ow, oh = (float(v) for v in other.get("box", [0, 0, 0, 0]))
        gap = max(0.0, max(y, oy) - min(y + h, oy + oh))
        if gap <= max(24.0, max(h, oh) * 1.15):
            return True
    return False


def annotate_native_lists(text_records: list[dict]) -> list[dict]:
    """Mark leading bullet/number text as native PPT list paragraphs.

    OCR sees bullets as literal glyphs, but PowerPoint stores them as
    paragraph properties with a hanging indent. Keeping them as normal
    text makes both font-size matching and x-position calibration treat
    the bullet gap as part of the word. The layout keeps the original
    text and char boxes for calibration, and adds a `list` spec so the
    PPT writer can strip the marker and emit native bullets/numbers.
    """
    candidates: list[tuple[int, dict, dict]] = []
    for idx, el in enumerate(text_records):
        prefix = _list_prefix(str(el.get("text") or ""))
        if prefix is not None:
            candidates.append((idx, el, prefix))
    if not candidates:
        return text_records

    for cand_idx, (_idx, el, prefix) in enumerate(candidates):
        geometry = _list_marker_geometry(el, int(prefix["marker_chars"]))
        if geometry is None:
            continue
        marker_x, body_x = geometry
        # A leading middle-dot can be real punctuation. Promote it to a
        # list item only when it has a neighbouring list row or when OCR
        # exposes a clear bullet-to-body hanging indent.
        has_indent = body_x - marker_x >= 6.0
        if not has_indent and not _has_list_neighbour(cand_idx, candidates):
            continue
        spec = {
            "kind": prefix["kind"],
            "marker_chars": int(prefix["marker_chars"]),
            "marker_text": prefix.get("marker_text"),
            "marker_x": int(round(marker_x)),
            "body_x": int(round(body_x)),
        }
        if prefix["kind"] == "bullet":
            spec["marker"] = prefix.get("marker", "•")
        else:
            spec["start"] = int(prefix.get("start", 1))
            spec["auto_type"] = prefix.get("auto_type", "arabicPeriod")
        el["list"] = spec
        el["valign"] = "top"
    return text_records


def strip_leading_list_markers(text_records: list[dict]) -> list[dict]:
    """Temporarily ignore leading bullet markers.

    Bullets/list markers need a unified pass that can combine OCR text,
    dot-shaped connected components, and neighbouring rows. Until that
    pass exists, do not let a leading `·`/`•` participate in font-size or
    position calibration. The marker is removed from editable text and
    target geometry is advanced to the body start when we can infer it.
    """
    for el in text_records:
        text = str(el.get("text") or "")
        prefix = _list_prefix(text)
        if not prefix or prefix.get("kind") != "bullet":
            continue
        marker_chars = int(prefix.get("marker_chars") or 0)
        if marker_chars <= 0:
            continue

        geometry = _list_marker_geometry(el, marker_chars)
        body_x = None
        if geometry is not None:
            _marker_x, body_x = geometry

        chars = el.get("source_chars")
        boxes = el.get("source_char_boxes")
        if chars and boxes and len(chars) == len(boxes):
            marker_boxes = [
                boxes[i] for i in range(min(marker_chars, len(boxes)))
                if str(chars[i]).strip() and len(boxes[i]) == 4
            ]
            if marker_boxes:
                el["ignored_marker_box"] = [
                    int(min(b[0] for b in marker_boxes)),
                    int(min(b[1] for b in marker_boxes)),
                    int(max(b[2] for b in marker_boxes)),
                    int(max(b[3] for b in marker_boxes)),
                ]
        elif geometry is not None:
            marker_x, _body_x = geometry
            bbox = el.get("source_bbox")
            if bbox and len(bbox) == 4:
                _x1, y1, _x2, y2 = (float(v) for v in bbox)
                h = max(1.0, y2 - y1)
                el["ignored_marker_box"] = [
                    int(round(marker_x - max(1.0, h * 0.20))),
                    int(round(y1)),
                    int(round(marker_x + max(2.0, h * 0.35))),
                    int(round(y2)),
                ]

        el["text"] = text[marker_chars:].lstrip()
        el.pop("list", None)

        runs = el.get("runs")
        if runs:
            remaining = marker_chars
            stripped: list[dict] = []
            leading_done = False
            for run in runs:
                r_text = str(run.get("text") or "")
                if remaining:
                    if len(r_text) <= remaining:
                        remaining -= len(r_text)
                        continue
                    r_text = r_text[remaining:]
                    remaining = 0
                if not leading_done:
                    r_text = r_text.lstrip()
                    leading_done = True
                if not r_text:
                    continue
                new_run = dict(run)
                new_run["text"] = r_text
                stripped.append(new_run)
            if stripped:
                el["runs"] = stripped
            else:
                el.pop("runs", None)

        if chars and boxes and len(chars) == len(boxes):
            # Drop the marker and any spaces immediately following it.
            drop = min(marker_chars, len(chars))
            while drop < len(chars) and not str(chars[drop]).strip():
                drop += 1
            el["source_chars"] = chars[drop:]
            el["source_char_boxes"] = boxes[drop:]
            if el["source_char_boxes"]:
                body_x = float(min(int(b[0]) for b in el["source_char_boxes"]))

        if body_x is not None:
            for key in ("source_bbox", "target_ink", "fit_target_ink"):
                value = el.get(key)
                if value and len(value) == 4:
                    value[0] = int(max(float(value[0]), float(body_x)))
            box = el.get("box")
            if box and len(box) == 4:
                old_x = float(box[0])
                shift = max(0.0, float(body_x) - old_x)
                # Keep a little left breathing room for antialiasing.
                shift = max(0.0, shift - 2.0)
                if shift:
                    box[0] = int(round(old_x + shift))
                    box[2] = int(round(max(1.0, float(box[2]) - shift)))
    return text_records


def _detect_leading_dot_marker(
    el: dict,
    source: np.ndarray,
) -> dict | None:
    """Find a small dot-like component just before a text line.

    This is intentionally colour-agnostic: bullets can be grey, blue,
    white, or any theme colour. False positives are controlled later by
    requiring repeated marker/body indentation across neighbouring rows.
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
            "body_x": float(body_x if body_x is not None else marker_box[2] + line_h),
            "line_y": float(line_y if line_y is not None else (marker_box[1] + marker_box[3]) / 2.0),
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
        gx1, gy1, gx2, gy2 = sx1 + cx, sy1 + cy, sx1 + cx + cw, sy1 + cy + ch
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


def _marker_has_neighbour(idx: int, candidates: list[tuple[dict, dict]]) -> bool:
    el, info = candidates[idx]
    for j, (_other_el, other) in enumerate(candidates):
        if j == idx:
            continue
        line_h = max(float(info.get("line_h") or 1.0),
                     float(other.get("line_h") or 1.0))
        body_close = abs(float(info["body_x"]) - float(other["body_x"])) <= max(12.0, line_h * 0.90)
        marker_close = abs(float(info["marker_x"]) - float(other["marker_x"])) <= max(10.0, line_h * 0.75)
        y_gap = abs(float(info["line_y"]) - float(other["line_y"]))
        if body_close and marker_close and y_gap <= max(48.0, line_h * 3.3):
            return True
    return False


def _marker_rgba_from_source(source: np.ndarray,
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
    alpha = np.clip((diff.astype(np.int16) - 8) * 14, 0, 255).astype(np.uint8)
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
        if bool(info.get("forced")) or _marker_has_neighbour(idx, candidates):
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


# =============================================================================
# Main: build manifest + layout, crop asset PNGs
# =============================================================================


def run(*, inventory: str, source: str, cleaned: str,
        asset_prefix: str, out_assets_dir: str,
        out_manifest: str, out_layout: str,
        slide_width_in: float | None = None,
        slide_height_in: float = 7.5) -> None:
    """Programmatic entry — see parse_args() for the CLI equivalent.

    Used by run_pipeline.process_page to skip subprocess overhead.
    """
    args = argparse.Namespace(
        inventory=inventory, source=source, cleaned=cleaned,
        asset_prefix=asset_prefix, out_assets_dir=out_assets_dir,
        out_manifest=out_manifest, out_layout=out_layout,
        slide_width_in=slide_width_in, slide_height_in=slide_height_in,
    )
    _run(args)


def main() -> None:
    _run(parse_args())


def _run(args: argparse.Namespace) -> None:
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    source = cv2.imread(args.source)
    cleaned = cv2.imread(args.cleaned)
    if source is None or cleaned is None:
        raise SystemExit("Could not load images.")

    # Slide dimensions: auto-derive width from source aspect so 4:3
    # inputs produce a 4:3 slide and 16:9 inputs produce 13.333×7.5.
    # The user can override via --slide-width-in.
    source_h_px, source_w_px = source.shape[:2]
    inpaint_scale = source_h_px / 720.0
    if args.slide_width_in is None:
        args.slide_width_in = args.slide_height_in * (source_w_px / source_h_px)
    # Resolution-independent calibration: every source pixel maps to
    # this many pt on the rendered slide. font_size_pt and any other
    # pixel-tuned heuristic should multiply by pt_per_px to stay
    # consistent across input resolutions.
    pt_per_px = (args.slide_height_in * 72.0) / source_h_px
    # Optional sidecar from build_inventory: text-erased but shapes intact.
    # Used as the asset crop source for subicon and internal-shape elements
    # (role: "source") so those crops don't carry duplicate OCR text that
    # would render on top of the editable text placed at the same spot.
    cleaned_path = Path(args.cleaned)
    text_only_path = cleaned_path.with_name(
        f"{cleaned_path.stem}.text_only.png")
    text_only = cv2.imread(str(text_only_path)) if text_only_path.exists() else None

    asset_dir = Path(args.out_assets_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    for f in asset_dir.glob("*.png"):
        f.unlink()

    manifest_assets = []
    image_elements: list[dict] = []
    shape_elements: list[dict] = []
    text_elements: list[dict] = []
    text_boxes = [
        tuple(int(v) for v in el["bbox"])
        for el in inventory
        if el.get("type") == "text"
        and str(el.get("text", "") or "").strip()
    ]
    image_inventory = [
        el for el in inventory
        if el.get("type") == "image" and len(el.get("bbox", [])) == 4
    ]
    outline_inventory = [
        el for el in image_inventory if el.get("role") == "outline"
    ]
    title_parent_candidates = [
        el for el in image_inventory
        if el.get("role") in {"outline", "background"}
    ]

    def _is_empty_asset(crop_bgr: np.ndarray,
                         alpha: np.ndarray | None,
                         *,
                         sparse_ok: bool = False) -> bool:
        """Return True when the asset has no visible content to render.

        Two failure modes we want to skip:
          - The crop sits on an erased region — almost every pixel is the
            slide bg colour (near-white), so the asset PNG is a blank
            rectangle on top of whatever's actually behind it.
          - We have an alpha mask, but every opaque pixel is also near-
            white. This happens when an upstream detector leaves a thin
            frame around an emptied card; the rendered PNG then looks like
            a washed-out rectangle on top of the real children below.

        "Near-white" = max channel diff vs white ≤ 16 — catches the
        anti-aliased card-bg tints (~#F6FAFD, ~#FAFCFF) typical of light
        UI cards in the source decks.
        """
        h, w = crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return True
        diff = np.abs(crop_bgr.astype(int) - 255).max(axis=2)
        near_white = diff <= 16
        if alpha is not None:
            opaque = alpha > 16
            opaque_count = int(opaque.sum())
            opaque_frac = opaque_count / float(h * w)
            if opaque_count == 0:
                return True
            if sparse_ok:
                # Thin card borders and line-art masks are intentionally
                # sparse. Do not drop them just because their opaque pixels
                # occupy <2% of the bbox; only skip true dust.
                return opaque_count < max(8, int(0.00015 * h * w))
            if opaque_frac < 0.02:
                return True
            # Among opaque pixels, how many are background-tinted? If the
            # mask is a thin frame around an erased card, this is ~100 %.
            opaque_white = int((near_white & opaque).sum())
            opaque_white_frac = opaque_white / float(opaque_count)
            # Big region segments (>30k px) get a pass even at high white-
            # frac: those are card panels whose visual identity is the
            # rounded corners + faint tint, not the interior fill.
            if opaque_white_frac >= 0.92 and (h * w) < 30000:
                return True
            return False
        near_white_frac = float(near_white.sum()) / float(h * w)
        visible = ~near_white
        # Sparse line art can be 95%+ white by bbox area but still be the
        # entire visible element. Keep it when there are enough meaningful non-bg
        # pixels; only skip crops that are truly blank or mere dust.
        if near_white_frac >= 0.95:
            elongated_rule = (
                (w >= 35 and h <= 4)
                or (h >= 35 and w <= 4)
            )
            if elongated_rule and int(diff.max()) > 5:
                return False
            median_bgr = np.median(crop_bgr.reshape(-1, 3), axis=0)
            median_diff_white = int(
                np.max(np.abs(median_bgr.astype(int) - 255))
            )
            # Large pale panels (#F4F8FF-ish) are visually meaningful even
            # though every pixel is "near white" by the blank-crop test.
            # Keep them; truly erased/blank regions have a median much
            # closer to pure white.
            if h * w >= 3000 and median_diff_white > 6:
                return False
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            meaningful = visible & ((gray < 235) | (hsv_[:, :, 1] > 18))
            meaningful_count = int(meaningful.sum())
            if meaningful_count >= max(80, int(0.001 * h * w)):
                return False
            return True
        if sparse_ok:
            visible_count = int(visible.sum())
            return visible_count < max(4, int(0.001 * h * w))
        return False

    child_roles = {"internal", "subicon", "line_subicon", "preserve_visual_icon"}

    def _child_source_image(child: dict) -> np.ndarray:
        if child.get("role") == "preserve_visual_icon":
            return source
        if child.get("source") == "source":
            return text_only if text_only is not None else source
        if child.get("source") == "original":
            return source
        return text_only if text_only is not None else cleaned

    def _child_inpaint_mask(child: dict) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        cx1, cy1, cx2, cy2 = (int(v) for v in child["bbox"])
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        src = _child_source_image(child)
        h_src, w_src = src.shape[:2]
        cx1 = max(0, min(w_src, cx1)); cx2 = max(0, min(w_src, cx2))
        cy1 = max(0, min(h_src, cy1)); cy2 = max(0, min(h_src, cy2))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop_child = src[cy1:cy2, cx1:cx2]
        role = child.get("role")
        mask_path = child.get("mask_path")
        if mask_path and Path(mask_path).exists():
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == src.shape[:2]:
                mask = m[cy1:cy2, cx1:cx2] > 16
                if int(mask.sum()) >= 4:
                    return (cx1, cy1, cx2, cy2), mask
        if role == "subicon":
            gray = cv2.cvtColor(crop_child, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop_child, cv2.COLOR_BGR2HSV)
            mask = (gray > 185) & (hsv_[:, :, 1] < 70)
        elif role == "line_subicon":
            mask = _line_art_alpha(crop_child) > 12
        elif role in {"internal", "preserve_visual_icon"}:
            mask = np.ones(crop_child.shape[:2], dtype=bool)
        else:
            gray = cv2.cvtColor(crop_child, cv2.COLOR_BGR2GRAY)
            hsv_ = cv2.cvtColor(crop_child, cv2.COLOR_BGR2HSV)
            diff_white = np.abs(crop_child.astype(int) - 255).max(axis=2)
            mask = ((gray < 245) & (diff_white > 5)) | (hsv_[:, :, 1] > 12)
            if int(mask.sum()) < 20:
                mask = np.ones(crop_child.shape[:2], dtype=bool)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=1).astype(bool)
        return (cx1, cy1, cx2, cy2), mask

    def _contained_child_entries(parent: dict,
                                 parent_box: tuple[int, int, int, int]) -> list[dict]:
        px1, py1, px2, py2 = parent_box
        p_area = max(1, (px2 - px1) * (py2 - py1))
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)
        out: list[dict] = []
        for child in image_inventory:
            if child is parent:
                continue
            role = child.get("role")
            if role in {"background", "outline"}:
                continue
            cx1, cy1, cx2, cy2 = (int(v) for v in child["bbox"])
            c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if c_area >= p_area * 0.55:
                continue
            cw, ch = cx2 - cx1, cy2 - cy1
            explicit_child = role in child_roles
            implicit_small_child = c_area <= 18000 and max(cw, ch) <= 180
            implicit_flat_child = (
                c_area <= 30000
                and min(cw, ch) <= 90
                and cw <= 0.95 * pw
                and ch <= 0.80 * ph
            )
            if not explicit_child and not implicit_small_child and not implicit_flat_child:
                continue
            ox1, oy1 = max(px1, cx1), max(py1, cy1)
            ox2, oy2 = min(px2, cx2), min(py2, cy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            if (ox2 - ox1) * (oy2 - oy1) >= 0.85 * c_area:
                out.append(child)
        return out

    def _inpaint_children_for_parent_asset(
        crop_bgr: np.ndarray,
        parent: dict,
        parent_box: tuple[int, int, int, int],
    ) -> np.ndarray:
        role = parent.get("role")
        if role in {"background", "internal", "subicon", "line_subicon"}:
            return crop_bgr
        children = _contained_child_entries(parent, parent_box)
        if not children:
            return crop_bgr
        px1, py1, px2, py2 = parent_box
        mask = np.zeros(crop_bgr.shape[:2], dtype=bool)
        for child in children:
            child_mask_info = _child_inpaint_mask(child)
            if child_mask_info is None:
                continue
            (cx1, cy1, cx2, cy2), child_mask = child_mask_info
            ox1, oy1 = max(px1, cx1), max(py1, cy1)
            ox2, oy2 = min(px2, cx2), min(py2, cy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            dst_x1, dst_y1 = ox1 - px1, oy1 - py1
            dst_x2, dst_y2 = ox2 - px1, oy2 - py1
            src_x1, src_y1 = ox1 - cx1, oy1 - cy1
            src_x2 = src_x1 + (dst_x2 - dst_x1)
            src_y2 = src_y1 + (dst_y2 - dst_y1)
            mask[dst_y1:dst_y2, dst_x1:dst_x2] |= child_mask[src_y1:src_y2, src_x1:src_x2]
        if int(mask.sum()) < 4:
            return crop_bgr
        patched = crop_bgr.copy()
        inpaint_region_inplace(patched, mask, radius=3, scale=inpaint_scale)
        return patched

    def _is_redundant_multi_card_outline(el: dict) -> bool:
        x1, y1, x2, y2 = (int(v) for v in el["bbox"])
        area = max(1, (x2 - x1) * (y2 - y1))
        contained_boxes: list[tuple[int, int, int, int]] = []
        for other in outline_inventory:
            if other is el:
                continue
            ox1, oy1, ox2, oy2 = (int(v) for v in other["bbox"])
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            if oarea >= area * 0.85:
                continue
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.72 * oarea:
                contained_boxes.append((ox1, oy1, ox2, oy2))
        if len(contained_boxes) < 2:
            return False
        # Redundant contour artefacts usually wrap two real cards placed
        # side-by-side (high vertical overlap, little horizontal overlap).
        # A true outer card can legitimately contain two stacked inner
        # panels; keep those.
        for i, a in enumerate(contained_boxes):
            ax1, ay1, ax2, ay2 = a
            aw, ah = ax2 - ax1, ay2 - ay1
            for b in contained_boxes[i + 1:]:
                bx1, by1, bx2, by2 = b
                bw, bh = bx2 - bx1, by2 - by1
                x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
                side_by_side = (
                    y_overlap >= 0.70 * min(ah, bh)
                    and x_overlap <= 0.20 * min(aw, bw)
                )
                if side_by_side:
                    return True
        return False

    def _outline_carried_by_large_parent(el: dict) -> bool:
        """Skip outline records already baked into a larger parent asset.

        Some residual parent crops intentionally preserve broad background
        structure such as connector lines, dashed brackets, and card frames.
        If we also render the same outline as a native shape, the frame gets
        visibly doubled. Keep the parent crop and drop the duplicate outline
        in that case.
        """
        ox1, oy1, ox2, oy2 = (int(v) for v in el["bbox"])
        outline_area = max(1, (ox2 - ox1) * (oy2 - oy1))
        slide_area = max(1, source.shape[0] * source.shape[1])
        for parent in image_inventory:
            if parent is el:
                continue
            role = parent.get("role")
            if role in {"outline", "background", "internal", "subicon", "line_subicon"}:
                continue
            px1, py1, px2, py2 = (int(v) for v in parent["bbox"])
            parent_area = max(1, (px2 - px1) * (py2 - py1))
            if parent_area > 0.85 * slide_area:
                continue
            if parent_area < 2.8 * outline_area:
                continue
            ix1, iy1 = max(ox1, px1), max(oy1, py1)
            ix2, iy2 = min(ox2, px2), min(oy2, py2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.92 * outline_area:
                return True
        return False

    def _outline_has_top_badge(el: dict) -> bool:
        """Return True when a small banner overlaps the outline's top edge.

        Native PowerPoint round-rect lines draw through later image/text in
        some renderers because the line is a separate vector stroke. When a
        colored tab sits on the top border, preserving the original alpha
        outline PNG is safer: the source crop already contains the correct
        occlusion around the badge.
        """
        ox1, oy1, ox2, oy2 = (int(v) for v in el["bbox"])
        ow = max(1, ox2 - ox1)
        oh = max(1, oy2 - oy1)

        h_img, w_img = source.shape[:2]
        sx1 = max(0, ox1 - 10)
        sx2 = min(w_img, ox2 + 10)
        sy1 = max(0, oy1 - 38)
        sy2 = min(h_img, oy1 + min(70, max(34, oh // 3)))
        if sx2 > sx1 and sy2 > sy1:
            band = source[sy1:sy2, sx1:sx2]
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            # The deck's title tabs are saturated dark blue rounded
            # rectangles. Detect them from the original pixels because some
            # slides crop the tab together with the illustration, so there
            # is no standalone badge element in the inventory.
            blue = (
                (hsv[:, :, 0] >= 88) & (hsv[:, :, 0] <= 124)
                & (hsv[:, :, 1] >= 70)
                & (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 230)
            ).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
            blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=1)
            cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < max(55, 0.18 * ow) or bw > 0.78 * ow:
                    continue
                if bh < 18 or bh > min(72, max(28, 0.35 * oh)):
                    continue
                ax1, ay1 = sx1 + bx, sy1 + by
                ax2, ay2 = ax1 + bw, ay1 + bh
                cx = (ax1 + ax2) / 2.0
                overlaps_top = ay1 <= oy1 + 36 and ay2 >= oy1 - 16
                inside_x = ox1 - 8 <= cx <= ox2 + 8
                if overlaps_top and inside_x:
                    return True

        for other in image_inventory:
            if other is el or other.get("role") in {"outline", "background"}:
                continue
            bx1, by1, bx2, by2 = (int(v) for v in other["bbox"])
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            if bw > 0.45 * ow or bh > 0.18 * oh:
                continue
            cx = (bx1 + bx2) / 2.0
            overlaps_top = by1 <= oy1 + 24 and by2 >= oy1 - 18
            inside_x = ox1 - 6 <= cx <= ox2 + 6
            if overlaps_top and inside_x:
                return True
        return False

    def _contained_child_outline_boxes(el: dict) -> list[tuple[int, int, int, int]]:
        """Return nested outline boxes carried by an outer outline mask."""
        ox1, oy1, ox2, oy2 = (int(v) for v in el["bbox"])
        outline_area = max(1, (ox2 - ox1) * (oy2 - oy1))
        boxes: list[tuple[int, int, int, int]] = []
        for other in image_inventory:
            if other is el or other.get("role") != "outline":
                continue
            cx1, cy1, cx2, cy2 = (int(v) for v in other["bbox"])
            child_area = max(1, (cx2 - cx1) * (cy2 - cy1))
            if child_area >= 0.85 * outline_area:
                continue
            ix1, iy1 = max(ox1, cx1), max(oy1, cy1)
            ix2, iy2 = min(ox2, cx2), min(oy2, cy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter >= 0.88 * child_area:
                boxes.append((cx1, cy1, cx2, cy2))
        return boxes

    def _outline_contains_child_outline(el: dict) -> bool:
        """Treat an outline with nested outline cards as a container only."""
        return bool(_contained_child_outline_boxes(el))

    for el in inventory:
        x1, y1, x2, y2 = el["bbox"]
        if el["type"] == "image":
            asset_name = f"{el['id']}.png"
            # Logo composites are cropped from the ORIGINAL image so brand
            # text inside the logo stays visible. Subicons (white-on-dark
            # pictograms detected inside dark cards) are also cropped from
            # the source AND get a transparency mask so only the white icon
            # strokes show — that lets the icon sit on top of the parent
            # card without the dark-blue square around it. Internal shapes
            # (numbered badges, photo placeholders) are source-cropped and
            # rendered as opaque rectangles. preserve_visual_icon crops come
            # from the original image too, but only after run_pipeline.py has
            # proven the segment is a small icon, so normal body text is not
            # baked into broad card assets. Other elements come from the
            # cleaned image (text already erased, sub-shape holes filled by
            # build_inventory).
            if el.get("role") == "preserve_visual_icon":
                src_img = source
            elif el.get("source") == "original":
                src_img = source
            elif el.get("source") == "source":
                # Prefer text-erased sidecar so the shape asset doesn't
                # carry duplicate OCR text. Fall back to source if the
                # sidecar is missing (older build_inventory runs).
                src_img = text_only if text_only is not None else source
            else:
                src_img = cleaned
            role = el.get("role")
            mask_path = el.get("mask_path")
            has_mask = bool(mask_path and Path(mask_path).exists())
            if role == "outline" and _outline_carried_by_large_parent(el):
                continue
            native_outline = (
                ENABLE_NATIVE_OUTLINE_SHAPES
                and
                role == "outline"
                and _outline_should_be_native_shape(
                    (int(x1), int(y1), int(x2), int(y2)))
                and not _outline_has_top_badge(el)
            )
            if native_outline:
                if _is_redundant_multi_card_outline(el):
                    continue
                line_color = _sample_outline_color(
                    source, (int(x1), int(y1), int(x2), int(y2)), mask_path)
                fill_color = _sample_card_fill_color(
                    source, (int(x1), int(y1), int(x2), int(y2)))
                shape_elements.append({
                    "type": "shape",
                    "name": el["id"],
                    "shape": "round_rect",
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "fill": fill_color,
                    "line": line_color,
                    "line_width": 0.9,
                    "radius": 0.08,
                })
                continue
            text_erased_crop = (
                src_img is cleaned
                or (text_only is not None and src_img is text_only)
            )
            original_w, original_h = int(x2 - x1), int(y2 - y1)
            implicit_cv2_icon = (
                role in {None, ""}
                and original_w <= 340
                and original_h <= 340
                and original_w * original_h <= 80000
            )
            sparse_visual = role in _SPARSE_VISUAL_ROLES or implicit_cv2_icon
            x1, y1, x2, y2 = _pad_icon_bbox(
                (x1, y1, x2, y2), src_img, text_boxes, role,
                allow_text_overlap=text_erased_crop)
            crop = src_img[y1:y2, x1:x2].copy()
            crop = _inpaint_children_for_parent_asset(
                crop, el, (int(x1), int(y1), int(x2), int(y2)))
            contained_outline_boxes = (
                _contained_child_outline_boxes(el)
                if role == "outline" else []
            )
            keep_outline_full_crop = (
                role == "outline"
                and not contained_outline_boxes
                and _outline_should_keep_full_crop(
                    source, (int(x1), int(y1), int(x2), int(y2)))
            )
            if keep_outline_full_crop:
                split_boxes = _split_filled_outline_rows(
                    source, (int(x1), int(y1), int(x2), int(y2)))
                if len(split_boxes) > 1:
                    for idx, split_box in enumerate(split_boxes):
                        sx1, sy1, sx2, sy2 = split_box
                        split_asset_name = f"{el['id']}_s{idx:02d}.png"
                        split_crop = src_img[sy1:sy2, sx1:sx2].copy()
                        split_crop = _inpaint_children_for_parent_asset(
                            split_crop, el, split_box)
                        if _is_empty_asset(split_crop, None, sparse_ok=sparse_visual):
                            continue
                        cv2.imwrite(str(asset_dir / split_asset_name), split_crop)
                        manifest_assets.append({
                            "name": split_asset_name,
                            "box": [int(sx1), int(sy1), int(sx2), int(sy2)],
                            "mode": "keep",
                        })
                        image_elements.append({
                            "type": "image",
                            "name": f"{el['id']}_s{idx:02d}",
                            "path": f"{args.asset_prefix}/{split_asset_name}",
                            "box": [int(sx1), int(sy1),
                                    int(sx2 - sx1), int(sy2 - sy1)],
                        })
                    continue
            if has_mask and not keep_outline_full_crop:
                # An upstream detector emitted a precise instance mask for
                # this element. Use it as the asset's alpha channel so the
                # PNG comes out with a true transparent background that
                # follows the segment's shape — no rectangular halo.
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape[:2] == src_img.shape[:2]:
                    alpha = m[y1:y2, x1:x2].copy()
                    for cx1, cy1, cx2, cy2 in contained_outline_boxes:
                        rx1 = max(0, cx1 - int(x1) - 3)
                        ry1 = max(0, cy1 - int(y1) - 3)
                        rx2 = min(alpha.shape[1], cx2 - int(x1) + 3)
                        ry2 = min(alpha.shape[0], cy2 - int(y1) + 3)
                        if rx2 > rx1 and ry2 > ry1:
                            alpha[ry1:ry2, rx1:rx2] = 0
                    if _is_empty_asset(crop, alpha, sparse_ok=sparse_visual):
                        continue
                    rgba = np.dstack([crop, alpha])
                    cv2.imwrite(str(asset_dir / asset_name), rgba)
                    manifest_assets.append({
                        "name": asset_name,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "mode": "keep",
                    })
                    image_elements.append({
                        "type": "image",
                        "name": el["id"],
                        "path": f"{args.asset_prefix}/{asset_name}",
                        "box": [int(x1), int(y1),
                                int(x2 - x1), int(y2 - y1)],
                    })
                    continue
            if _is_empty_asset(crop, None, sparse_ok=sparse_visual):
                continue
            if el.get("role") == "subicon":
                # White icon on dark bg → keep only near-white pixels opaque,
                # everything else transparent. Threshold matches the
                # detect_white_subicons hole criterion (luminance > 220,
                # saturation < 40), but per-pixel so antialiased edges fade
                # via the alpha channel instead of hard-cutting.
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                hsv_ = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                alpha = np.clip((gray.astype(int) - 180) * 3, 0, 255).astype(np.uint8)
                alpha[hsv_[:, :, 1] > 60] = 0
                rgba = np.dstack([crop, alpha])
                cv2.imwrite(str(asset_dir / asset_name), rgba)
            elif el.get("role") == "line_subicon":
                alpha = _line_art_alpha(crop)
                if int((alpha > 16).sum()) < 4:
                    continue
                rgba = np.dstack([crop, alpha])
                cv2.imwrite(str(asset_dir / asset_name), rgba)
            else:
                cv2.imwrite(str(asset_dir / asset_name), crop)
            manifest_assets.append({
                "name": asset_name,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "mode": "keep",
            })
            image_elements.append({
                "type": "image",
                "name": el["id"],
                "path": f"{args.asset_prefix}/{asset_name}",
                "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            })
        else:  # text
            if not str(el.get("text", "") or "").strip():
                continue
            x1, y1, x2, y2 = _trim_short_stat_text_bbox(
                str(el.get("text", "") or ""),
                (int(x1), int(y1), int(x2), int(y2)),
                source,
            )
            # PaddleOCR per-char bboxes when present. ocr_paddle attaches
            # these only when len(chars) == len(text), and we re-verify
            # against the current text (it may have been corrected by
            # ocr_review_apply, in which case alignment is gone and we
            # fall back to proportional estimation inside detect_text_style).
            raw_text = el["text"]
            safe_text = ppt_safe_text(raw_text)
            cb = el.get("char_boxes")
            if cb is not None and len(cb) != len(raw_text):
                cb = None
            wd = el.get("words")
            wb = el.get("word_boxes")
            if (wd is not None and wb is not None
                    and (len(wd) != len(wb) or "".join(wd) != raw_text)):
                wd = wb = None
            source_char_boxes = _derive_source_char_boxes(
                raw_text, (int(x1), int(y1), int(x2), int(y2)),
                cb, wd, wb)
            style = detect_text_style([x1, y1, x2, y2], source,
                                      text=raw_text,
                                      char_boxes=source_char_boxes,
                                      words=wd, word_boxes=wb)
            # Convert glyph_h (measured ink extent per word) into per-run
            # font size. α=0.78 same as the line-level formula, applied
            # against measured ink height rather than bbox height —
            # gives values that line up with the line's own size when
            # ink is comparable (e.g. `7` in `7分钟` should land near
            # the line's 36 pt, not balloon to 60 pt off a too-narrow
            # word_box that exaggerates a stroke's vertical extent).
            apply_run_sizes = _should_apply_run_font_sizes(raw_text)
            if "runs" in style:
                for r in style["runs"]:
                    gh = r.pop("glyph_h", None)
                    if apply_run_sizes and gh and gh > 0:
                        raw = gh * pt_per_px * CAPTURE_FACTOR_SHORT
                        r["size"] = _snap_to_ladder(max(8.0, min(36.0, raw)))
                if apply_run_sizes:
                    # Same-colour runs in one record should share a font
                    # size — `升级为`=14 pt next to `软约束`=16 pt looks
                    # ragged. Group by colour, pick the mode size in each
                    # group, and snap runs within ±2 ladder steps to it.
                    _unify_run_sizes_by_color(style["runs"])
                    # Punctuation typographically inherits the em-box of
                    # its neighbouring text — a `"` between 16 pt body runs
                    # should also render at 16 pt, even though the quote
                    # glyph itself only occupies ~8 px.
                    _inherit_punct_sizes(style["runs"])
            mixed_size = None
            if not apply_run_sizes:
                mixed_size = mixed_size_runs_from_char_boxes(
                    raw_text, source, (int(x1), int(y1), int(x2), int(y2)),
                    source_char_boxes, style.get("runs"), style["color"],
                    bold=bool(style["bold"]),
                    pt_per_px=pt_per_px,
                )
                if mixed_size is not None:
                    style["runs"] = mixed_size["runs"]
            if "runs" in style:
                for r in style["runs"]:
                    r["text"] = ppt_safe_text(r.get("text", ""))
            # Decoration-padding probe: when a multi-char line's ink
            # vertical extent is well under bbox_h (≤ 50 %), the bbox
            # likely includes shadow/glow above and below the actual
            # glyphs. Use ink_h instead. Guards:
            #   - text must have ≥ 3 visible chars — averages out the
            #     low-ink-glyph variance (`一` alone would otherwise be
            #     flagged and shrink to ~3 pt);
            #   - bbox_h must be > 30 px so we don't trip on tiny labels
            #     where ink-row measurement is noise-dominated.
            ink_h = style.pop("ink_h", None)
            effective_h = int(y2 - y1)
            visible_chars = sum(1 for c in raw_text
                                if c.strip() and c != "\n")
            if (ink_h and effective_h > 30 and visible_chars >= 3
                    and ink_h <= 0.5 * effective_h):
                effective_h = ink_h
            size = font_size_pt(safe_text, x2 - x1, effective_h,
                                pt_per_px=pt_per_px)
            if mixed_size is not None:
                size = max(int(size), int(mixed_size["base_size"]))
            ppt_font = default_ppt_font()
            text_bold = bool(style["bold"])
            run_sized = (
                any(r.get("size") is not None
                    for r in style.get("runs", []))
            )
            compact_text = "".join(c for c in safe_text if not c.isspace())
            bbox_w = max(1, int(x2 - x1))
            bbox_h = max(1, int(y2 - y1))
            badge_like_number = (
                compact_text.isdigit()
                and len(compact_text) <= 3
                and bbox_h >= 24
                and 0.65 <= (bbox_w / float(bbox_h)) <= 1.60
            )
            render_fit = None
            if not run_sized and not badge_like_number:
                render_fit = fit_text_render(
                    safe_text, source, (int(x1), int(y1), int(x2), int(y2)),
                    initial_size=int(size),
                    initial_bold=text_bold,
                    initial_font=ppt_font,
                    pt_per_px=pt_per_px,
                )
                if render_fit is not None:
                    size = int(render_fit["size"])
                    text_bold = bool(render_fit["bold"])
            # Tight bbox: do NOT inflate the text box beyond what OCR found.
            # A loose box with left-align makes the text appear shifted from
            # its source position. Just a 2 px right + 2 px bottom safety
            # margin keeps a slightly larger render from clipping.
            align = "left"
            # Left edge anchored to OCR x1; 6 px right + 2 px bottom buffers
            # for font-metric variance (without them glyphs that render
            # slightly wider than the OCR bbox wrap to a second visual line
            # even with word_wrap = False).
            base_w = int(x2 - x1)
            text_w = max(
                base_w + 6,
                _estimated_text_width_px(
                    safe_text, int(size),
                    pt_per_px=pt_per_px,
                    bold=text_bold,
                ) + 8,
            )
            if run_sized:
                text_w = max(
                    text_w,
                    _estimated_runs_width_px(
                        style.get("runs", []), int(size),
                        pt_per_px=pt_per_px,
                        bold=text_bold,
                    ) + 8,
                )
            text_h = int(y2 - y1) + 2
            if run_sized:
                text_h = max(
                    text_h,
                    int(round((int(size) / max(0.01, pt_per_px)) * 1.20)),
                )
            text_x = int(x1)
            text_y = int(y1)
            valign_mode = "middle"
            if align == "center" and text_w > base_w + 6:
                center = (int(x1) + int(x2)) / 2.0
                text_x = int(round(center - text_w / 2.0))
                text_x = max(0, min(source.shape[1] - text_w, text_x))
            if render_fit is not None and align != "center":
                text_x, text_y, text_w, text_h = render_fit["box"]
                valign_mode = "top"
            elif mixed_size is not None:
                valign_mode = "top"
            # All title centring runs at PPTX-build time in
            # build_pptx_from_layout.apply_title_centering, which
            # filters on length ≤ 8 + smallest-containing-image +
            # centre-tolerance.
            record = {
                "type": "text",
                "name": el["id"],
                "text": safe_text,
                "box": [text_x, text_y, text_w, text_h],
                "source_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "font": ppt_font,
                "size": int(size),
                "bold": text_bold,
                "color": style["color"],
                "align": align,
                "valign": valign_mode,
                "line_spacing": 1.0,
            }
            if (source_char_boxes is not None
                    and len(source_char_boxes) == len(raw_text)
                    and len(safe_text) == len(raw_text)):
                record["source_chars"] = list(safe_text)
                record["source_char_boxes"] = [
                    [int(v) for v in box] for box in source_char_boxes
                ]
            target_metrics = _target_text_metrics(
                source, (int(x1), int(y1), int(x2), int(y2)))
            if target_metrics is not None:
                record["target_ink"] = target_metrics["ink_bbox"]
            if render_fit is not None:
                record["size_source"] = "render_fit"
                record["font_fit_score"] = render_fit["score"]
                record["render_fit_font"] = render_fit["font"]
                record["fit_target_ink"] = render_fit["target_ink"]
                record["fit_render_ink"] = render_fit["render_ink"]
            elif mixed_size is not None:
                record["size_source"] = "mixed_runs"
                record["mixed_size"] = {
                    "base_size": mixed_size["base_size"],
                    "suffix_size": mixed_size["suffix_size"],
                    "prefix_h": mixed_size["prefix_h"],
                    "suffix_h": mixed_size["suffix_h"],
                    "prefix_bold": mixed_size.get("prefix_bold"),
                    "suffix_bold": mixed_size.get("suffix_bold"),
                    "prefix_density": mixed_size.get("prefix_density"),
                    "prefix_fit_score": mixed_size["prefix_fit_score"],
                    "suffix_fit_score": mixed_size["suffix_fit_score"],
                }
            # Optional in-bbox per-character color runs. Only present when
            # the line has more than one detected color group; consumers
            # (build_pptx_from_layout) should fall back to `color` when
            # `runs` is missing.
            if "runs" in style:
                record["runs"] = style["runs"]
            text_elements.append(record)

    # Multi-line merging is intentionally DISABLED. Heuristics on column
    # overlap + style similarity can over-merge unrelated stacked text on
    # dense slides. The merge function is kept in this
    # file for reuse; the call site is just commented out — to re-enable,
    # uncomment the line below.
    # text_elements = merge_multiline_texts(text_elements)
    text_elements = strip_leading_list_markers(text_elements)
    # Native list conversion is intentionally not enabled yet. Leading
    # bullet glyphs are ignored above; a later unified pass should recover
    # bullets from OCR + connected components and then emit PPT list
    # paragraph properties consistently.
    # text_elements = annotate_native_lists(text_elements)
    image_elements.extend(
        restore_ignored_bullet_marker_images(
            text_elements, source, asset_dir, args.asset_prefix)
    )
    # Size unification still runs — it just normalises font-size drift
    # between visually-identical labels (`1000亿元` vs `24.22%`).
    text_elements = unify_group_sizes(text_elements)

    # Topo-sort images by bbox containment so a parent card never paints
    # over an icon nested inside it. The role-priority hack in
    # build_inventory.py (background → outline → internal → subicon) only
    # covers role-labelled cases; this catches every same-role parent/
    # child pair (e.g. two `foreground` cards nested).
    image_elements = topo_sort_by_containment(image_elements)

    manifest = {
        "source": str(Path(args.cleaned).absolute()),
        "assets": manifest_assets,
    }
    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    h, w = source.shape[:2]
    layout = {
        "slide_size": {"width_in": args.slide_width_in, "height_in": args.slide_height_in},
        "source_width": w,
        "source_height": h,
        "background": estimate_slide_background_hex(source),
        # Native card/frame conversion is optional and currently disabled;
        # outline/card visuals are emitted as images to avoid renderer-added
        # strokes crossing title tabs.
        "elements": shape_elements + image_elements + text_elements,
    }
    Path(args.out_layout).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_layout).write_text(json.dumps(layout, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps({
        "text_elements": len(text_elements),
        "image_elements": len(image_elements),
        "shape_elements": len(shape_elements),
        "manifest": str(args.out_manifest),
        "layout": str(args.out_layout),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
