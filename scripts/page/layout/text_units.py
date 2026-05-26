"""Per-character width units + derived char-box helpers + width estimates.

When PaddleOCR does not return a per-char or per-word box (or the text
was OCR-review-corrected and bboxes no longer align), proportional width
units (CJK 1.0, ASCII 0.56, …) slice the line bbox into character-level
pieces so per-char sampling can still operate. Width estimates feed the
sizing/positioning logic — they predict how wide a given size+text will
render so the layout writes a wide-enough text frame.
"""
from __future__ import annotations


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


def _text_width_units(text: str) -> float:
    units = 0.0
    for c in text or "":
        units += _char_width_unit(c)
    return units


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
    """Estimate source-pixel width needed to avoid PPT text wrapping."""
    lines = str(text or "").split("\n")
    max_units = max((_text_width_units(line) for line in lines),
                    default=0.0)
    if max_units <= 0 or pt_per_px <= 0:
        return 0
    weight = 1.08 if bold else 1.0
    # 1.10 absorbs LibreOffice/PowerPoint font metric differences
    # without making normal left-aligned OCR boxes visibly drift.
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
