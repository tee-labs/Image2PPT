"""Bold-flag majority vote across visually-similar text records.

Sizes are NOT touched here: per-bbox `font_size_pt(text, bbox_w, bbox_h)`
already derives the size directly from the OCR bbox geometry, and that
is treated as authoritative. The pass denoises only the bold flag (which
is derived from stroke-density and is noisy per-bbox for thin glyphs
like `%`).
"""
from __future__ import annotations

from layout.color import _color_close
from layout.text_sizing import _is_stat_value


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
            # bbox height within ±25 % of the seed.
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
