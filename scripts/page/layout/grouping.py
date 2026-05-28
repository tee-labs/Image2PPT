"""Bold + font-family majority vote across visually-similar text records.

Sizes are NOT touched here: per-bbox `font_size_pt(text, bbox_w, bbox_h)`
already derives the size directly from the OCR bbox geometry, and that
is treated as authoritative. The pass denoises:

- the bold flag (derived from stroke density, noisy on thin glyphs like `%`)
- the font family + italic, propagating any high-confidence ML font
  prediction in the group to its low-confidence neighbours so that
  visually-similar elements (same paragraph, same colour, same height)
  end up with a consistent font in the final PPTX.
"""
from __future__ import annotations

from collections import Counter

from layout.color import _color_close
from layout.text_sizing import _is_stat_value


# Predictions with confidence >= this value are trusted enough to vote on
# behalf of their bucket. Picked above the per-element apply threshold (0.60)
# to make sure only fairly confident neighbours influence others.
_VOTE_CONF_FLOOR = 0.60


def unify_group_sizes(text_records: list[dict]) -> list[dict]:
    """Group by category + fuzzy-color + similar bbox height, then take
    a majority bold vote inside each group.

    Categories:
      - `stat`: numeric values with units (`1000亿元`, `24.22%`)
      - `short`, `medium`, `long`: by char count (mirrors font_size_pt)

    Colour comparison is fuzzy (per-channel ±25) so anti-aliasing
    variants end up in the same group. bbox height similarity prevents
    a cover title (h≈150) from clustering with same-colour info-row
    text (h≈30).
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

    def has_cjk(text: str) -> bool:
        # CJK Unified Ideographs (covers Chinese/Japanese kanji/Hangul-adjacent
        # ranges that show up in typical PPT decks)
        for c in text:
            o = ord(c)
            if 0x4E00 <= o <= 0x9FFF:
                return True
            if 0x3400 <= o <= 0x4DBF:  # CJK Ext A
                return True
            if 0xF900 <= o <= 0xFAFF:  # CJK Compatibility
                return True
        return False

    groups: list[list[int]] = []
    assigned = [False] * len(text_records)
    for i, r in enumerate(text_records):
        if assigned[i]:
            continue
        cat_i = category(r["text"])
        col_i = r["color"]
        h_i = r["box"][3]
        cjk_i = has_cjk(r["text"])
        bucket = [i]
        assigned[i] = True
        for j, other in enumerate(text_records):
            if assigned[j]:
                continue
            if category(other["text"]) != cat_i:
                continue
            if not _color_close(other["color"], col_i):
                continue
            # bbox height within ±25 % of the seed.
            h_j = other["box"][3]
            if max(h_i, h_j) > min(h_i, h_j) * 1.25:
                continue
            # Don't mix CJK text with pure-ASCII / math glyphs. They share
            # colour and height bands but live in different font families
            # (e.g. Cambria-italic math symbols vs Microsoft YaHei CJK).
            if has_cjk(other["text"]) != cjk_i:
                continue
            bucket.append(j)
            assigned[j] = True
        groups.append(bucket)

    # Majority vote per group on bold, plus font/italic propagation driven
    # by any high-confidence ML predictions in the bucket. Size is left
    # untouched — font_size_pt(bbox_w, bbox_h) already derived it from the
    # authoritative PaddleOCR bbox.
    out = [dict(r) for r in text_records]
    for bucket in groups:
        if len(bucket) < 2:
            continue

        # Bold majority vote.
        bold_votes = sum(1 for i in bucket if out[i]["bold"])
        unified_bold = bool(bold_votes * 2 >= len(bucket))
        for i in bucket:
            out[i]["bold"] = unified_bold

        # Font family + italic propagation. Only fire when the bucket has
        # at least one confidently-predicted member: that anchor's family
        # gets pushed to neighbours whose own prediction was below the
        # apply threshold (and therefore fell back to the default font).
        confident = [
            i for i in bucket
            if (out[i].get("font_pred") or {}).get("family_confidence", 0.0)
            >= _VOTE_CONF_FLOOR
        ]
        if not confident:
            continue
        fam_counts = Counter(out[i]["font_pred"]["family"] for i in confident)
        # italic was applied to the record itself by text_emit when the
        # prediction was confident, so we read it from the top-level field.
        ital_counts = Counter(bool(out[i].get("italic", False))
                              for i in confident)
        winning_family = fam_counts.most_common(1)[0][0]
        winning_italic = ital_counts.most_common(1)[0][0]
        confident_set = set(confident)
        for i in bucket:
            if i in confident_set:
                # Honour the element's own confident prediction; don't let
                # group majority overwrite a high-conf disagreement (e.g.
                # a math symbol predicted Cambria sitting in a CJK bucket).
                continue
            out[i]["font"] = winning_family
            out[i]["italic"] = winning_italic
    return out
