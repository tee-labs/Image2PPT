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
from pathlib import Path

import cv2
import numpy as np


ENABLE_NATIVE_OUTLINE_SHAPES = False


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
    if b > r + 20 and b > 100:
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

    Rule: for each pure-punctuation run, copy the size of the nearer
    non-punctuation neighbour (prefer the larger when both exist).
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
        for j in range(i - 1, -1, -1):
            if not is_punct[j] and runs[j].get("size") is not None:
                prev_size = int(runs[j]["size"])
                break
        next_size = None
        for j in range(i + 1, len(runs)):
            if not is_punct[j] and runs[j].get("size") is not None:
                next_size = int(runs[j]["size"])
                break
        if prev_size is None and next_size is None:
            continue
        runs[i]["size"] = max(s for s in (prev_size, next_size)
                              if s is not None)


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
    return (
        any(c.isdigit() for c in text)
        and len(text) <= 8
        and any(s in text for s in ["亿", "%", "个", "PB", "MB", "GB", "TB", "万"])
    )


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
    """Alpha mask for sparse coloured line-art cropped from text_clean."""
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
    keep = (diff > 16) & ((hsv[:, :, 1] > 16) | (gray < 238))
    alpha = np.clip((diff.astype(int) - 10) * 7, 0, 255).astype(np.uint8)
    alpha[~keep] = 0
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


# =============================================================================
# Main: build manifest + layout, crop asset PNGs
# =============================================================================


def main() -> None:
    args = parse_args()
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    source = cv2.imread(args.source)
    cleaned = cv2.imread(args.cleaned)
    if source is None or cleaned is None:
        raise SystemExit("Could not load images.")

    # Slide dimensions: auto-derive width from source aspect so 4:3
    # inputs produce a 4:3 slide and 16:9 inputs produce 13.333×7.5.
    # The user can override via --slide-width-in.
    source_h_px, source_w_px = source.shape[:2]
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
        inpaint_mask = mask.astype(np.uint8) * 255
        return cv2.inpaint(np.ascontiguousarray(crop_bgr), inpaint_mask, 3,
                           cv2.INPAINT_TELEA)

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
            cb = el.get("char_boxes")
            if cb is not None and len(cb) != len(el["text"]):
                cb = None
            wd = el.get("words")
            wb = el.get("word_boxes")
            if (wd is not None and wb is not None
                    and (len(wd) != len(wb) or "".join(wd) != el["text"])):
                wd = wb = None
            style = detect_text_style([x1, y1, x2, y2], source,
                                      text=el["text"], char_boxes=cb,
                                      words=wd, word_boxes=wb)
            # Convert glyph_h (measured ink extent per word) into per-run
            # font size. α=0.78 same as the line-level formula, applied
            # against measured ink height rather than bbox height —
            # gives values that line up with the line's own size when
            # ink is comparable (e.g. `7` in `7分钟` should land near
            # the line's 36 pt, not balloon to 60 pt off a too-narrow
            # word_box that exaggerates a stroke's vertical extent).
            if "runs" in style:
                for r in style["runs"]:
                    gh = r.pop("glyph_h", None)
                    if gh and gh > 0:
                        raw = gh * pt_per_px * CAPTURE_FACTOR_SHORT
                        r["size"] = _snap_to_ladder(max(8.0, min(36.0, raw)))
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
            visible_chars = sum(1 for c in el["text"]
                                if c.strip() and c != "\n")
            if (ink_h and effective_h > 30 and visible_chars >= 3
                    and ink_h <= 0.5 * effective_h):
                effective_h = ink_h
            size = font_size_pt(el["text"], x2 - x1, effective_h,
                                pt_per_px=pt_per_px)
            # Tight bbox: do NOT inflate the text box beyond what OCR found.
            # A loose box with left-align makes the text appear shifted from
            # its source position. Just a 2 px right + 2 px bottom safety
            # margin keeps a slightly larger render from clipping.
            short_value = (
                len(el["text"]) <= 6
                and any(c.isdigit() for c in el["text"])
            )
            align = "center" if short_value else "left"
            # Left edge anchored to OCR x1; 6 px right + 2 px bottom buffers
            # for font-metric variance (without them glyphs that render
            # slightly wider than the OCR bbox wrap to a second visual line
            # even with word_wrap = False).
            text_w = int(x2 - x1) + 6
            text_h = int(y2 - y1) + 2
            # All title centring runs at PPTX-build time in
            # build_pptx_from_layout.apply_title_centering, which
            # filters on length ≤ 8 + smallest-containing-image +
            # centre-tolerance.
            record = {
                "type": "text",
                "name": el["id"],
                "text": el["text"],
                "box": [int(x1), int(y1), text_w, text_h],
                "font": "Microsoft YaHei",
                "size": int(size),
                "bold": bool(style["bold"]),
                "color": style["color"],
                "align": align,
                "valign": "middle",
                "line_spacing": 1.0,
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
        "background": "#FFFFFF",
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
