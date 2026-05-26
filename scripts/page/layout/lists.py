"""Leading bullet / ordered-list marker handling.

Bullets and numbered prefixes confuse the size/position calibration
because OCR sees them as glyphs whose stroke shape is unlike normal
text. This module recognises them, records the marker bbox so it can be
restored as an image, and strips the marker from the editable text so
downstream calibration measures only the body.
"""
from __future__ import annotations

import re


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
        if (suffix in {".", "．"} and match.end() < len(text)
                and text[match.end()].isdigit()):
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


def _list_marker_geometry(el: dict,
                          marker_chars: int) -> tuple[float, float] | None:
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


def strip_leading_list_markers(text_records: list[dict]) -> list[dict]:
    """Remove leading bullet markers from text + advance target geometry.

    Bullets/list markers need a unified pass that combines OCR text,
    dot-shaped connected components, and neighbouring rows. Until that
    pass exists, a leading `·`/`•` is not allowed to participate in
    font-size or position calibration. The marker is removed from
    editable text and target geometry is advanced to the body start.
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
                body_x = float(
                    min(int(b[0]) for b in el["source_char_boxes"]))

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
