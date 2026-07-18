"""Per-character / per-word colour run detection inside one OCR text bbox.

PaddleOCR returns one bbox per text line; when a line has mixed colours,
the line-level median washes the variation to one dominant colour. The
single function exported here recovers it by sampling each character (or
word) inside the bbox and grouping consecutive same-coloured segments
into runs.
"""
from __future__ import annotations

import numpy as np

from layout.color import (
    _color_close,
    _hex_to_rgb,
    _sampled_color_hex,
)


def _per_char_runs(bbox: list[int], text: str, orig_img: np.ndarray,
                   bg: np.ndarray, fallback_color: str,
                   char_boxes: list[list[int]] | None = None,
                   words: list[str] | None = None,
                   word_boxes: list[list[int]] | None = None) -> list[dict]:
    """Detect in-bbox colour variation and group same-coloured segments.

    Three sampling modes (preference order):

      C) Per-WORD bboxes from PaddleOCR (`words` + `word_boxes` from
         return_word_box=True). PP-OCRv6 segments by character class
         (continuous digits as one word, each CJK glyph as one word),
         so `524个` arrives as words=['524','个'] with two distinct
         bboxes — exactly the granularity at which colour changes
         typically happen. Used when len(words) ≥ 2.

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
    gives the char colour. Consecutive chars within ±25 per channel are
    grouped into one run.

    Returns a list of {"text": str, "color": "#RRGGBB"} dicts. The
    caller decides whether to emit a `runs` field (only when len > 1).
    """
    x1, y1, x2, y2 = bbox
    region = orig_img[y1:y2, x1:x2]
    bbox_w = x2 - x1
    if region.size == 0 or bbox_w <= 0 or len(text) <= 1:
        return [{"text": text, "color": fallback_color}]

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
        # part: `524个`, `7分钟`, `20.26%`). PP-OCRv6's word_box
        # boundaries align with the visual change there. Pure CJK
        # lines have each glyph as a one-char word — Mode C there
        # would devolve into noisy per-char colour from antialiasing,
        # so they fall through to Mode A.
        h_img, w_img = orig_img.shape[:2]
        # PP-OCRv6 occasionally returns overlapping word_boxes (e.g. a
        # `分` box extending into the next `钟`'s ink). Clip each
        # box's edges against neighbours so samples capture only the
        # intended glyph.
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
                    raw = _sampled_color_hex(sub[mask], bg)
                    col = (fallback_color
                           if _color_close(raw, fallback_color, tol=60)
                           else raw)
                    # Glyph height = vertical extent of stroke rows
                    # with ≥ 2 ink px. Per-row count filter rejects
                    # single antialias pixels.
                    row_has_ink = mask.sum(axis=1) >= 2
                    ink_rows = np.where(row_has_ink)[0]
                    if ink_rows.size >= 2:
                        glyph_h = int(ink_rows.max() - ink_rows.min() + 1)
            per_word.append({"text": w, "color": col, "glyph_h": glyph_h})
        # Two-pass merge:
        #   1. Snap each entry's raw colour to a running cluster
        #      centroid within ±40 channels (anti-alias tolerance).
        #   2. Merge consecutive entries with same clustered colour
        #      AND glyph height within 30 % — different-size segments
        #      stay separate (`7分钟`: 7 tall, 分钟 small → 2 runs).
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
            cx1 = max(0, int(cb[0]))
            cy1 = max(0, int(cb[1]))
            cx2 = min(w_img, int(cb[2]))
            cy2 = min(h_img, int(cb[3]))
            # Inset by 1 px on each side to avoid antialiased edge
            # bleed between adjacent characters.
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
            col = _sampled_color_hex(pixels, bg)
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
            sx1, sx2 = (cx_start + margin,
                        max(cx_start + margin + 1, cx_end - margin))
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
            col = _sampled_color_hex(pixels, bg)
            char_colors.append((c, col))

    # Anchor on the line-level dominant colour: any per-char sample
    # close to dominant snaps to dominant. Two-tier closeness:
    #   - Standard: ±60 per channel — handles most anti-aliasing.
    #   - Gray-aware: when BOTH dominant and sampled are near-neutral,
    #     allow up to 130 luminance difference for two near-greys
    #     before treating it as a real colour jump.
    fr, fg, fb = _hex_to_rgb(fallback_color)
    fallback_is_gray = (abs(fr - fg) < 20 and abs(fg - fb) < 20
                       and abs(fr - fb) < 20)
    fallback_lum = 0.299 * fr + 0.587 * fg + 0.114 * fb
    for i, (c, col) in enumerate(char_colors):
        if col is None or col == fallback_color:
            continue
        if _color_close(col, fallback_color, tol=60):
            char_colors[i] = (c, fallback_color)
            continue
        if fallback_is_gray:
            cr, cg, cb = _hex_to_rgb(col)
            col_is_gray = (abs(cr - cg) < 20 and abs(cg - cb) < 20
                          and abs(cr - cb) < 20)
            if col_is_gray:
                col_lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
                if abs(col_lum - fallback_lum) < 130:
                    char_colors[i] = (c, fallback_color)

    # Group consecutive same-coloured chars. ±40 fallback merges
    # adjacent runs that landed at slightly different hexes. Chars that
    # failed sampling (None) inherit the current run.
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

    if not runs:
        return [{"text": text, "color": fallback_color}]

    # Suppress single-character non-dominant runs — they are typically
    # per-char sampling noise. Real emphasis usually covers ≥ 2 chars.
    def _visible_char_count(t: str) -> int:
        return sum(1 for c in t if c.strip())

    cleaned: list[dict] = []
    for r in runs:
        if (r["color"] != fallback_color
                and _visible_char_count(r["text"]) < 2):
            if cleaned:
                cleaned[-1]["text"] += r["text"]
            else:
                cleaned.append({"text": r["text"], "color": fallback_color})
        else:
            if cleaned and cleaned[-1]["color"] == r["color"]:
                cleaned[-1]["text"] += r["text"]
            else:
                cleaned.append(r)
    return cleaned
