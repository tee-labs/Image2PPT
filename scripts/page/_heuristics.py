"""OCR-item-level heuristics shared by erase_text + build_inventory + run_pipeline.

These functions decide what to do with each PaddleOCR detection BEFORE
any pixel work:

  - `should_preserve_visual` — was this item flagged by the review
    stream (or by an icon-internal label) to stay as pixels?
  - `is_likely_icon` — does this look like an icon misclassified as
    text (low-conf single char, square edge density, sits inside a
    small colored container, etc.)? Returns the rule that fired so
    gray-zone calls can be routed into the LLM icon-review packet.
  - `detect_logo_strips` / `is_in_logo_zone` — the auto-detection
    layer for brand-strip preservation. Currently disabled; logo
    preservation is whitelist-driven via `find_template_logos.py`.
    The API is kept so callers don't need to change.
  - `preprocess_ocr` (with `split_icon_prefix` and
    `trim_ocr_bbox_off_trailing_visual`) — cleans the OCR bboxes
    BEFORE erase: strips row-leading icon prefixes,
    trims trailing/leading off-colour visuals that PaddleOCR pulled
    into a text bbox (stat-callout arrows, status pips).

Every consumer (erase_text.py for the actual erasure, build_inventory
for the candidate-text filter, run_pipeline for the icon-decision
flow) imports from this module, so the OCR-level verdicts are
unambiguous across the pipeline.

This module is a library — no CLI, no main.
"""
from __future__ import annotations

import cv2
import numpy as np


# =============================================================================
# Resolution-scale helpers
# =============================================================================
# Every pixel threshold in this module + build_inventory + icon/* was tuned
# on 720-tall source slides. Callers compute `scale = h_img / 720.0` (via
# `pixel_scale`) and route each length/area/kernel constant through these
# helpers so a 1080p input (scale=1.5) filters 2.25× larger areas, uses
# ~1.5× longer dimensions, and applies proportionally bigger morphology
# kernels — without retuning the constants by hand.
#
# The reference height is the source height the constants were tuned at;
# scale=1.0 (i.e. h_img=720) reproduces the original behaviour exactly.
REFERENCE_HEIGHT_PX = 720


def pixel_scale(img_or_h) -> float:
    """Compute `h / 720.0` from either a numpy image or an int height."""
    if isinstance(img_or_h, np.ndarray):
        return float(img_or_h.shape[0]) / float(REFERENCE_HEIGHT_PX)
    return float(img_or_h) / float(REFERENCE_HEIGHT_PX)


def s_length(v: float, scale: float) -> int:
    """Scale a 720-tuned pixel length; floors to 1 to never collapse a real
    dimension to 0."""
    return max(1, int(round(v * scale)))


def s_area(v: float, scale: float) -> int:
    """Scale a 720-tuned pixel area (length²); floors to 1."""
    return max(1, int(round(v * scale * scale)))


def s_kernel(v: float, scale: float) -> int:
    """Scale a 720-tuned morphology kernel. Floors to 1; does NOT snap to
    odd — cv2.morphologyEx accepts even kernel sizes fine, and forcing
    odd would change the baseline (e.g. the default 6×6 dilate kernel)."""
    return max(1, int(round(v * scale)))


# =============================================================================
# Per-item preservation flag
# =============================================================================


def should_preserve_visual(item: dict) -> bool:
    """OCR item should remain as pixels, not editable text.

    Review corrections use an empty string to mean "not real editable
    text". For icon-internal glyphs such as `￥`, that must not imply
    erasing the glyph from the source image.
    """
    text = str(item.get("text", "") or "").strip()
    return bool(item.get("preserve_visual")) or text == ""


def _looks_like_decorative_binary(text: str, cw: int, ch: int) -> bool:
    """Binary-code glyphs often live inside data/globe icons.

    OCR reads them as normal text, but erasing them hollows out the icon.
    Keep this deliberately narrow so table values like `0`/`1` still become
    editable text.
    """
    compact = text.replace(" ", "").replace("…", "").replace(".", "")
    if len(compact) < 4 or not compact:
        return False
    if any(c not in "01" for c in compact):
        return False
    return ch <= 30 and cw <= 130


# =============================================================================
# Logo zone (auto-detection retired; API preserved)
# =============================================================================


def detect_logo_strips(ocr_data: list[dict], image_height: int) -> list[tuple]:
    """Logo-strip auto-detection is intentionally disabled.

    The previous heuristic (>=4 short texts in a bottom-of-slide y-band)
    fired on legitimate footer banners too. Some process diagrams and
    slogan footers share the same shape and had their text baked into a
    "logo zone", making it un-editable.

    Logo / brand preservation is now whitelist-driven:
      - `find_template_logos.py --whitelist <file>` (mechanism C) marks
        exact text strings as `preserve_visual`.
      - `find_template_logos.py` mechanism B still auto-detects logos
        by cross-page repeat — it can't fire on a single page and only
        matches when the same text recurs in the same bbox across many
        pages, so it doesn't risk eating per-page content.
      - The OCR-review step can mark any specific OCR item with
        `preserve_visual: true` if needed.

    The is_in_logo_zone API is preserved (always returns False now) so
    callers don't need to change. Returns an empty band list.
    """
    return []


def is_in_logo_zone(item: dict, logo_bands: list[tuple]) -> bool:
    cx = (item["x1"] + item["x2"]) / 2
    cy = (item["y1"] + item["y2"]) / 2
    return any(y1 <= cy <= y2 and x1 <= cx <= x2 for y1, y2, x1, x2, _ in logo_bands)


# =============================================================================
# OCR bbox cleanup (prefix split + trailing-visual trim)
# =============================================================================


def split_icon_prefix(item: dict) -> dict:
    """When OCR includes a row-leading icon glyph in the text bbox (e.g. it
    reads a symbol, separator, then label text), trim the icon portion
    from the text content and SHRINK the bbox to exclude that area.

    This is a SPLIT, not a merge — opposite of what we removed earlier.
    The icon area is left untouched in the source image, so the icon stays
    visible in the cleaned image as its own visual component.

    Trigger: a `|` or `丨` separator within the first 4 characters AND text
    has at least one CJK or alphanumeric character after the separator.
    The shift amount is one bbox-height (one character width) on the left.
    """
    text = str(item.get("text", "") or "")
    if len(text) < 2:
        return item
    sep_pos = -1
    for i, c in enumerate(text[:4]):
        if c in "|丨":
            sep_pos = i
            break
    if sep_pos < 0:
        return item
    new_text = text[sep_pos + 1:].strip()
    # Bail out if there is nothing meaningful after the separator.
    if not new_text:
        return item
    # Require post-separator text to contain at least one real character so
    # we don't accidentally strip a meaningful expression.
    if not any(c.isalnum() or "一" <= c <= "鿿" for c in new_text):
        return item
    x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
    bh = y2 - y1
    icon_w = max(bh, 25)
    if x1 + icon_w + 10 >= x2:
        return item  # too narrow to split safely
    return {**item, "x1": x1 + icon_w + 4, "text": new_text}


def trim_ocr_bbox_off_trailing_visual(item: dict,
                                      img: np.ndarray) -> dict:
    """Trim an OCR bbox that extends past the actual glyphs into an
    adjacent visual.

    PaddleOCR sometimes pulls a small status-pip or growth-arrow icon
    into the same bbox as the stat-callout text (`524个` followed by a
    gray ↑ icon, `29 PB` followed by a status dot). The arrow is a
    *different* element — different colour, separated by a clear column
    gap — and should be preserved as its own visual asset. Today the
    over-wide bbox leaves the icon's gray pixels inside the text
    region: erase keeps them (they don't lie on the bg→text colour
    axis), build_inventory doesn't claim them either, and the PPT
    rendered text overlaps them.

    Detection (right side, symmetric for left):
      1. Sample foreground pixels per column inside the bbox.
      2. Group columns into glyph blobs separated by >=3 px bg gaps.
      3. The dominant text colour comes from the largest blob.
      4. Drop trailing/leading blobs whose median colour is more than
         60 (max-channel) away from the dominant colour. These almost
         always belong to a different element.
      5. Trim the bbox to the surviving blobs.

    Returns a possibly-modified copy of the OCR item. Leaves it
    untouched when the bbox shape doesn't fit this pattern (too tall,
    fewer than 2 blobs, dominant colour can't be determined).
    """
    text = str(item.get("text", "") or "").strip()
    if not text or item.get("preserve_visual") or item.get("_force_text"):
        return item
    x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
    if x2 - x1 < 30 or y2 - y1 < 12:
        return item
    h_img, w_img = img.shape[:2]
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w_img, x2)
    y2c = min(h_img, y2)
    if x2c - x1c < 30 or y2c - y1c < 12:
        return item
    region = img[y1c:y2c, x1c:x2c]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    fg = (gray < 235) | (hsv[:, :, 1] > 30)
    col_fg = fg.sum(axis=0)
    blobs: list[tuple[int, int]] = []
    start = None
    gap_run = 0
    for c in range(region.shape[1]):
        has_fg = col_fg[c] > 0
        if has_fg:
            if start is None:
                start = c
            gap_run = 0
        else:
            if start is not None:
                gap_run += 1
                if gap_run >= 3:
                    blobs.append((start, c - gap_run))
                    start = None
                    gap_run = 0
    if start is not None:
        blobs.append((start, region.shape[1] - 1))
    if len(blobs) < 2:
        return item
    # Compute per-blob median colour from foreground pixels.
    blob_info: list[tuple[int, int, int, np.ndarray]] = []
    for cs, ce in blobs:
        sub = region[:, cs:ce + 1]
        submask = fg[:, cs:ce + 1]
        pix_count = int(submask.sum())
        if pix_count == 0:
            continue
        fg_pix = sub[submask]
        med = np.median(fg_pix, axis=0)
        blob_info.append((cs, ce, pix_count, med))
    if len(blob_info) < 2:
        return item
    # Dominant text colour = median of the heaviest blob.
    dom = max(blob_info, key=lambda b: b[2])[3]
    # A blob is a "stray visual" to trim only when it sits at an edge
    # AND meets three conditions vs the median text blob:
    #   - colour off the text colour by > 60 (max-channel)
    #   - pixel count < 40 % of the median surviving blob
    #   - separated from its neighbour by >= 5 cols of bg
    # The size and gap checks reject false positives where an OCR run
    # really is multi-coloured text: the off-colour body sits right next
    # to the dominant run and is comparable in size, so the trim leaves
    # it alone.
    pix_counts = [b[2] for b in blob_info]
    pix_counts.sort()
    median_pix = pix_counts[len(pix_counts) // 2]

    def col_gap_before(i: int) -> int:
        if i == 0:
            return blob_info[i][0]
        return blob_info[i][0] - blob_info[i - 1][1] - 1

    def col_gap_after(i: int) -> int:
        if i == len(blob_info) - 1:
            return region.shape[1] - 1 - blob_info[i][1]
        return blob_info[i + 1][0] - blob_info[i][1] - 1

    def is_stray(i: int, side: str) -> bool:
        col_diff = int(np.max(np.abs(blob_info[i][3] - dom)))
        if col_diff <= 60:
            return False
        if blob_info[i][2] >= 0.4 * median_pix:
            return False
        gap = col_gap_before(i) if side == "left" else col_gap_after(i)
        return gap >= 5

    keep_mask = [True] * len(blob_info)
    for i in range(len(blob_info) - 1, -1, -1):
        if is_stray(i, "right"):
            keep_mask[i] = False
        else:
            break
    for i in range(len(blob_info)):
        if is_stray(i, "left"):
            keep_mask[i] = False
        else:
            break
    if all(keep_mask):
        return item
    surviving = [b for b, k in zip(blob_info, keep_mask) if k]
    if not surviving:
        return item
    new_x1 = x1c + surviving[0][0]
    new_x2 = x1c + surviving[-1][1] + 1
    if new_x1 >= new_x2 or (new_x1 == item["x1"] and new_x2 == item["x2"]):
        return item
    return {**item, "x1": int(new_x1), "x2": int(new_x2)}


def preprocess_ocr(ocr_data: list[dict],
                   img: np.ndarray | None = None) -> list[dict]:
    """Apply OCR cleanups:
      - Split row-leading icon glyphs off the text bbox
      - Trim trailing/leading visuals (off-colour neighbours)

    `img` is needed for the colour-based trim; when None, only the
    prefix split runs.
    """
    out = [split_icon_prefix(item) for item in ocr_data]
    if img is not None:
        out = [trim_ocr_bbox_off_trailing_visual(it, img) for it in out]
    return out


# =============================================================================
# Icon-vs-text classifier
# =============================================================================


def is_likely_icon(item: dict, all_items: list[dict],
                   img: np.ndarray) -> tuple[bool, str | None]:
    """Heuristics: skip OCR results that are likely icons misclassified as
    text. Returns (is_icon, reason) — reason names the rule that fired so
    callers can route uncertain calls (everything except `preserve_visual`
    and `decorative_binary`) into the LLM icon-review packet.

    The LLM icon-review stage injects two flags into the OCR item dict to
    override the heuristic deterministically across every call site
    (erase, build_inventory, run_pipeline's text-mask check):

    - `_force_text: True` -> reviewer flipped a preserved-as-icon call to
      text. Skip every icon rule so the item flows through as editable
      text.
    - `preserve_visual: True` -> reviewer confirmed/promoted to icon.
      Handled by `should_preserve_visual` below.
    """
    if item.get("_force_text"):
        return False, None
    if should_preserve_visual(item):
        return True, "preserve_visual"
    text = str(item.get("text", "") or "")
    conf = item.get("confidence", 1.0)
    cw = item["x2"] - item["x1"]
    ch = item["y2"] - item["y1"]
    # Every pixel constant below was tuned at 720-tall source; scale so a
    # 1080p input (scale=1.5) filters proportionally larger areas/lengths.
    scale = pixel_scale(img)
    if _looks_like_decorative_binary(text, cw, ch):
        return True, "decorative_binary"
    # Low confidence short text
    if len(text) <= 2 and conf < 0.6:
        return True, "low_conf_short_text"
    # Isolated single char
    if len(text) == 1:
        cy = (item["y1"] + item["y2"]) / 2
        cx = (item["x1"] + item["x2"]) / 2
        if not any(
            abs((o["y1"] + o["y2"]) / 2 - cy) < s_length(15, scale)
            and abs((o["x1"] + o["x2"]) / 2 - cx) < s_length(100, scale)
            for o in all_items
            if o is not item
        ):
            return True, "isolated_single_char"
    # Square aspect with high edge density
    if len(text) <= 2 and 0.7 < cw / ch < 1.4 and min(cw, ch) > s_length(30, scale):
        gray = cv2.cvtColor(img[item["y1"]:item["y2"], item["x1"]:item["x2"]], cv2.COLOR_BGR2GRAY)
        ed = cv2.Canny(gray, 50, 150).sum() / (cw * ch * 255)
        if ed > 0.15:
            return True, "high_edge_density"
    # Short glyph inside a light-background outlined icon. This catches
    # symbols like `￥` inside a circle/hexagon: the glyph bbox itself has
    # white bg, so the colored-region rule below cannot see the container.
    if len(text) <= 2 and max(cw, ch) <= s_length(45, scale):
        h, w = img.shape[:2]
        pad = max(s_length(8, scale), int(max(cw, ch) * 0.8))
        x1 = max(0, item["x1"] - pad)
        y1 = max(0, item["y1"] - pad)
        x2 = min(w, item["x2"] + pad)
        y2 = min(h, item["y2"] + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            fg = (gray < 235) | (hsv[:, :, 1] > 28)
            ix1 = max(0, item["x1"] - x1 - 1)
            iy1 = max(0, item["y1"] - y1 - 1)
            ix2 = min(fg.shape[1], item["x2"] - x1 + 1)
            iy2 = min(fg.shape[0], item["y2"] - y1 + 1)
            if ix2 > ix1 and iy2 > iy1:
                side_hit_threshold = s_area(5, scale)
                outer_hit_threshold = s_area(18, scale)
                side_hits = 0
                if int(fg[:iy1, ix1:ix2].sum()) >= side_hit_threshold:
                    side_hits += 1
                if int(fg[iy2:, ix1:ix2].sum()) >= side_hit_threshold:
                    side_hits += 1
                if int(fg[iy1:iy2, :ix1].sum()) >= side_hit_threshold:
                    side_hits += 1
                if int(fg[iy1:iy2, ix2:].sum()) >= side_hit_threshold:
                    side_hits += 1
                outer = fg.copy()
                outer[iy1:iy2, ix1:ix2] = False
                if side_hits >= 3 and int(outer.sum()) >= outer_hit_threshold:
                    return True, "outlined_icon_glyph"
    # Text inside SMALL square colored region (icon-internal labels)
    pad = s_length(5, scale)
    samples = []
    h, w = img.shape[:2]
    if item["y1"] > pad:
        samples.append(img[item["y1"] - pad:item["y1"], item["x1"]:item["x2"]])
    if item["y2"] + pad < h:
        samples.append(img[item["y2"]:item["y2"] + pad, item["x1"]:item["x2"]])
    if samples:
        bg = np.median(np.concatenate([s.reshape(-1, 3) for s in samples if s.size > 0]), axis=0)
        if not np.all(bg > 220):
            diff = np.abs(img.astype(int) - bg).max(axis=2)
            region = (diff < 30).astype(np.uint8) * 255
            n, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
            cy = (item["y1"] + item["y2"]) // 2
            sample_offset = s_length(3, scale)
            if item["x1"] > sample_offset:
                lbl = labels[cy, item["x1"] - sample_offset]
                if 0 < lbl < n:
                    bw, bh = stats[lbl, 2], stats[lbl, 3]
                    bw_min, bw_max = s_length(30, scale), s_length(250, scale)
                    bh_min, bh_max = bw_min, bw_max
                    if bw_min < bw < bw_max and bh_min < bh < bh_max and 0.5 < bw / bh < 2.0:
                        # The colored region must HUG the text on both axes —
                        # a tight badge does (ratios ~2x in both directions),
                        # a larger card does not (text is one element among
                        # several inside the card, so the card extends well
                        # past the text in at least one direction). The 3.0x
                        # bound rejects card-as-icon over-flagging for short
                        # labels inside larger cards.
                        if bw <= cw * 3.0 and bh <= ch * 3.0:
                            return True, "small_colored_container"
    return False, None


# Reasons that name a definitive decision: the rule is unambiguous and the
# LLM reviewer would be a no-op. `preserve_visual` is an explicit
# instruction from the OCR review stream; `decorative_binary` is a
# narrowly scoped icon-internal pattern. Every other reason is routed into
# the review packet.
DEFINITIVE_ICON_REASONS = {"preserve_visual", "decorative_binary"}
