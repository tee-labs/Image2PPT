#!/usr/bin/env python
"""Local table structure recognition for slide images.

Wraps PaddleOCR's SLANet_plus table-structure model (Baidu PaddlePaddle
team, open-source — same model family that backs Baidu's table OCR API).
Runs entirely on CPU, total model footprint ~8 MB. Cell text is read by
matching the structure cells against an OCR JSON (re-using the OCR
pipeline that already runs upstream — no extra model needed).

The agent identifies a table region visually and calls this script with
the slide image plus a bbox. SLANet_plus is a structure-only model: it
assumes its input IS a table and returns its best guess at the grid. If
you feed it a non-table region it will hallucinate cells. Don't.

Pipeline:
    slide image + bbox  ──▶  crop  ──▶  SLANet_plus (~0.1s, local)
                                            │
                                            ▼
                                  cell polygons + HTML structure
                                            │
                       (match against ocr.json by bbox containment)
                                            │
                                            ▼
                                       tables.json

Output schema (one element — this script processes one table region per
invocation; call N times for N tables on a slide):
    {
      "bbox": [x1, y1, x2, y2],         // table region in SOURCE-image coords
      "rows": 4, "cols": 5,
      "score": 0.98,                    // SLANet structure confidence
      "html": "<table>...</table>",
      "cells": [
        {"row": 0, "col": 0,
         "rowspan": 1, "colspan": 5,
         "bbox": [x1, y1, x2, y2],      // in SOURCE-image coords
         "text": "..."}
        ...
      ]
    }

Usage:
    # Agent already cropped the table to its own PNG
    python scripts/tables/table_recognize.py table_crop.png > table.json

    # Or: agent points at table region within the slide
    python scripts/tables/table_recognize.py slide.jpg \\
        --bbox 100,200,900,650 \\
        --ocr output/<run_dir>/inventory/page_NN.ocr.json \\
        --out output/<run_dir>/inventory/page_NN.table.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local table structure recognition (SLANet_plus)."
    )
    parser.add_argument("image",
                        help="Slide image (with --bbox) OR a pre-cropped "
                             "table image.")
    parser.add_argument("--bbox",
                        help="Optional crop region 'x1,y1,x2,y2' in source "
                             "image pixel coords. If omitted, the whole "
                             "image is treated as the table.")
    parser.add_argument("--ocr",
                        help="OCR JSON for the WHOLE slide (output of "
                             "ocr_paddle.py / ocr_vision.swift). Used to "
                             "fill cell text by bbox containment. If omitted "
                             "cells have empty text.")
    parser.add_argument("--out",
                        help="Output JSON path. Default: stdout.")
    parser.add_argument("--model",
                        default="SLANet_plus",
                        help="Table structure model name. Default SLANet_plus "
                             "(~8MB). Alternatives: SLANeXt_wired / "
                             "SLANeXt_wireless (~350MB each, higher accuracy "
                             "on wired / wireless tables respectively).")
    return parser.parse_args()


def quiet_paddle() -> None:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("FLAGS_print_log", "0")


def _poly_to_aabb(poly) -> tuple[int, int, int, int]:
    """Convert an 8-coord polygon [x1,y1,x2,y2,x3,y3,x4,y4] (or 4 (x,y)
    points) to an axis-aligned bbox."""
    flat = list(poly)
    if flat and isinstance(flat[0], (list, tuple)):
        xs = [int(p[0]) for p in flat]
        ys = [int(p[1]) for p in flat]
    else:
        xs = [int(flat[i]) for i in range(0, len(flat), 2)]
        ys = [int(flat[i]) for i in range(1, len(flat), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def _parse_structure(tokens: list[str]) -> list[dict]:
    """Walk SLANet structure tokens and assign (row, col, rowspan, colspan)
    to each cell, in the same order as the bbox list.

    SLANet emits HTML tokens. Cells appear as either:
      - compact: ``'<td></td>'`` (empty cell, no attrs)
      - expanded: ``'<td'`` then optional `' colspan="N"'` / `' rowspan="N"'`
        attribute fragments then ``'>'`` then ``'</td>'``
    """
    cells: list[dict] = []
    grid: list[list[bool]] = []  # occupancy by (row, col)
    r = -1
    i = 0
    while i < len(tokens):
        tk = tokens[i]
        if tk == "<tr>":
            r += 1
            if len(grid) <= r:
                grid.append([])
            i += 1
            continue
        if tk == "</tr>":
            i += 1
            continue
        if tk == "<td></td>" or tk == "<td>":
            rs = cs = 1
            j = i + 1
            # If it was '<td>' (open only), skip until '</td>'.
            if tk == "<td>":
                while j < len(tokens) and tokens[j] != "</td>":
                    j += 1
                j += 1
            i = j
        elif tk == "<td":
            rs = cs = 1
            j = i + 1
            while j < len(tokens) and tokens[j] != ">":
                m = re.search(r'rowspan="?(\d+)"?', tokens[j])
                if m:
                    rs = int(m.group(1))
                m = re.search(r'colspan="?(\d+)"?', tokens[j])
                if m:
                    cs = int(m.group(1))
                j += 1
            j += 1  # past '>'
            while j < len(tokens) and tokens[j] != "</td>":
                j += 1
            i = j + 1
        else:
            i += 1
            continue

        # Find leftmost free column in current row.
        if r < 0:
            r = 0
            grid.append([])
        c = 0
        while c < len(grid[r]) and grid[r][c]:
            c += 1
        # Mark occupancy.
        for dr in range(rs):
            while len(grid) <= r + dr:
                grid.append([])
            while len(grid[r + dr]) <= c + cs - 1:
                grid[r + dr].append(False)
            for dc in range(cs):
                grid[r + dr][c + dc] = True

        cells.append({
            "row": r, "col": c,
            "rowspan": rs, "colspan": cs,
        })
    return cells


def _reassemble_html(tokens: list[str]) -> str:
    return "".join(tokens)


def _attach_text_from_ocr(cell_bbox_abs, ocr_data: list[dict]) -> str:
    """Concatenate OCR text whose bbox center falls inside the cell."""
    x1, y1, x2, y2 = cell_bbox_abs
    hits = []
    for it in ocr_data:
        cx = (int(it["x1"]) + int(it["x2"])) // 2
        cy = (int(it["y1"]) + int(it["y2"])) // 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            hits.append(it)
    if not hits:
        return ""
    # Reading order: top-to-bottom, left-to-right.
    hits.sort(key=lambda it: (int(it["y1"]), int(it["x1"])))
    return " ".join(h["text"] for h in hits).strip()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        sys.stderr.write(f"ERROR: image not found: {image_path}\n")
        return 1

    img = Image.open(image_path).convert("RGB")
    if args.bbox:
        try:
            x1, y1, x2, y2 = (int(v) for v in args.bbox.split(","))
        except ValueError:
            sys.stderr.write(
                "ERROR: --bbox must be 'x1,y1,x2,y2' integers.\n"
            )
            return 1
        crop = img.crop((x1, y1, x2, y2))
        crop_offset = (x1, y1)
        table_bbox_abs = [x1, y1, x2, y2]
    else:
        crop = img
        crop_offset = (0, 0)
        table_bbox_abs = [0, 0, img.width, img.height]

    quiet_paddle()
    try:
        from paddlex import create_model
    except ImportError:
        sys.stderr.write(
            "ERROR: paddlex not installed. Run: "
            "pip install 'paddleocr>=3' 'paddlex[ocr]'\n"
        )
        return 1

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        crop.save(tf.name)
        crop_path = tf.name
    try:
        model = create_model(model_name=args.model)
        results = list(model.predict(crop_path))
    finally:
        try:
            os.unlink(crop_path)
        except OSError:
            pass

    if not results:
        sys.stderr.write("[table_recognize] No structure returned.\n")
        out = {}
    else:
        r0 = results[0]
        data = r0.json if hasattr(r0, "json") else r0
        payload = data.get("res", data) if isinstance(data, dict) else {}

        tokens = payload.get("structure") or []
        polys = payload.get("bbox") or []
        score = float(payload.get("structure_score") or 0.0)

        cells = _parse_structure(tokens)

        # Align polys with cells. Each <td...> consumed one bbox in the
        # order they appear. If counts mismatch (SLANet sometimes outputs
        # a bbox for the first whole-row header that's then merged) we
        # truncate to the shorter list and warn.
        if len(polys) != len(cells):
            sys.stderr.write(
                f"[table_recognize] WARN: {len(polys)} bboxes vs "
                f"{len(cells)} cells — truncating to "
                f"{min(len(polys), len(cells))}.\n"
            )
        n = min(len(polys), len(cells))

        ocr_data = []
        if args.ocr:
            ocr_data = json.loads(Path(args.ocr).read_text(encoding="utf-8"))

        ox, oy = crop_offset
        for i in range(n):
            x1, y1, x2, y2 = _poly_to_aabb(polys[i])
            abs_bbox = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
            cells[i]["bbox"] = abs_bbox
            cells[i]["text"] = _attach_text_from_ocr(abs_bbox, ocr_data) \
                if ocr_data else ""

        cells = cells[:n]
        rows = (max((c["row"] + c["rowspan"] for c in cells), default=0)) \
            if cells else 0
        cols = (max((c["col"] + c["colspan"] for c in cells), default=0)) \
            if cells else 0

        out = {
            "bbox": table_bbox_abs,
            "rows": rows,
            "cols": cols,
            "score": score,
            "html": _reassemble_html(tokens),
            "cells": cells,
        }

    out_json = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json, encoding="utf-8")
        sys.stderr.write(
            f"[table_recognize] {out.get('rows', 0)}x{out.get('cols', 0)} table "
            f"(score={out.get('score', 0):.3f}) → {out_path}\n"
        )
    else:
        print(out_json)
        sys.stderr.write(
            f"[table_recognize] {out.get('rows', 0)}x{out.get('cols', 0)} table "
            f"(score={out.get('score', 0):.3f})\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
