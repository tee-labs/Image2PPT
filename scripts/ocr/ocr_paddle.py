#!/usr/bin/env python
"""PaddleOCR PP-OCRv5 wrapper — single-page or batch.

This is the ONLY OCR backend in the skill. It reads slide images and
emits axis-aligned text bboxes in source-image pixel coordinates as a
JSON list of `{text, x1, y1, x2, y2, confidence}` objects. Downstream
`erase_text.py`, `build_inventory.py`, `inventory_to_layout.py` consume
this shape.

Single-page mode (writes JSON to stdout):
    python scripts/ocr/ocr_paddle.py slide.jpg > ocr.json

Batch mode (reuses ONE warm PaddleOCR model across all pages — the
~3s startup cost is paid once instead of N times):
    python scripts/ocr/ocr_paddle.py --batch \
        page_01.png  out/page_01.ocr.json \
        page_02.webp out/page_02.ocr.json \
        ...

PaddleOCR-VL 1.5 was evaluated and rejected: 80s+ per page on CPU,
only worth it with GPU acceleration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PP-OCRv5 OCR → JSON.")
    parser.add_argument("inputs", nargs="*",
                        help="Single mode: one image path. "
                             "Batch mode (with --batch): pairs of "
                             "image_path out_json_path.")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode — `inputs` is read as "
                             "(image, out_json) pairs and JSON is "
                             "written to each out_json instead of stdout.")
    parser.add_argument("--lang", default="ch",
                        help="Recognition language (default: ch).")
    parser.add_argument("--min-conf", type=float, default=0.3,
                        help="Drop detections below this confidence.")
    return parser.parse_args()


def quiet_paddle() -> None:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("FLAGS_print_log", "0")


def poly_to_bbox(poly) -> tuple[int, int, int, int]:
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def extract_items(result, min_conf: float) -> list[dict]:
    items: list[dict] = []
    if not result:
        return items
    r0 = result[0]
    data = r0.json if hasattr(r0, "json") else r0
    payload = data.get("res", data) if isinstance(data, dict) else r0
    boxes = payload.get("rec_boxes")
    polys = payload.get("rec_polys") or payload.get("dt_polys") or []
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or [1.0] * len(texts)
    # Per-WORD data — populated when PaddleOCR is initialised with
    # return_word_box=True (see PaddleOCR(...) below). These are parallel
    # to texts/scores: text_words[i] is a list of "words" for line i,
    # where PP-OCRv5 segments by character class (continuous digits as
    # one word, continuous CJK as another). E.g. `63个` → ['63', '个']
    # with two parallel boxes; `数智互联` → ['数智互联'] with one box.
    # Carried through to ocr.json so downstream can:
    #   - sample per-segment colors for in-bbox color changes (red
    #     keyword inside black sentence),
    #   - distinguish mixed-size segments like `63个` (big digits + small
    #     CJK unit) — though PP-OCRv5 currently returns identical y
    #     extents for all words on a line, so this is width-only info.
    text_words = payload.get("text_word") or []
    text_word_boxes = payload.get("text_word_boxes") or []
    for i, (text, score) in enumerate(zip(texts, scores)):
        if float(score) < min_conf or not text or not text.strip():
            continue
        if boxes is not None and i < len(boxes):
            bx = boxes[i]
            x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        else:
            x1, y1, x2, y2 = poly_to_bbox(polys[i])
        item = {
            "text": text,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": float(score),
        }
        # Attach per-segment data (PP-OCRv5 emits "words" = consecutive
        # same-class runs: each CJK glyph alone, digits clustered). Two
        # exposures:
        #   - words/word_boxes: always, when joined equal text. Lets
        #     downstream see e.g. `63个` as ['63', '个'] for mixed-size
        #     handling.
        #   - chars/char_boxes: only when each word is a single
        #     character (pure CJK lines). Preserves the legacy per-char
        #     interface used by detect_text_style for color sampling.
        if i < len(text_words) and i < len(text_word_boxes):
            words = list(text_words[i])
            word_boxes_i = text_word_boxes[i]
            if (len(words) == len(word_boxes_i)
                    and "".join(words) == text):
                box_ints = [
                    [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                    for b in word_boxes_i
                ]
                item["words"] = words
                item["word_boxes"] = box_ints
                if all(len(w) == 1 for w in words):
                    item["chars"] = words
                    item["char_boxes"] = box_ints
        items.append(item)
    return items


def main() -> int:
    args = parse_args()
    quiet_paddle()
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        sys.stderr.write(
            "ERROR: paddleocr not installed. Run: "
            "pip install 'paddleocr>=3' 'paddlex[ocr]'\n"
        )
        return 1

    # Disable the document preprocessor stack — for slide images those
    # reshape the input image and report bboxes in the deskewed coordinate
    # system, which does NOT match the source pixel grid.
    #
    # return_word_box=True asks PP-OCRv5 to also emit per-character bboxes
    # alongside the line-level boxes. Downstream uses these to detect
    # in-bbox color changes (e.g. a red keyword inside a black line).
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )

    if args.batch:
        if len(args.inputs) % 2 != 0:
            sys.stderr.write("ERROR: --batch expects an even number of "
                             "arguments (image, out_json pairs).\n")
            return 2
        pairs = [(Path(args.inputs[i]), Path(args.inputs[i + 1]))
                 for i in range(0, len(args.inputs), 2)]
        for img_path, out_path in pairs:
            if not img_path.exists():
                sys.stderr.write(f"SKIP missing: {img_path}\n")
                continue
            try:
                result = ocr.predict(str(img_path))
            except Exception as exc:
                sys.stderr.write(f"FAIL {img_path}: {exc}\n")
                continue
            items = extract_items(result, args.min_conf)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"ok {img_path.name} -> {out_path.name} "
                  f"({len(items)} items)", flush=True)
        return 0

    if len(args.inputs) != 1:
        sys.stderr.write("ERROR: single mode expects exactly one image "
                         "path; use --batch for multi-page input.\n")
        return 2
    image_path = Path(args.inputs[0])
    if not image_path.exists():
        sys.stderr.write(f"ERROR: image not found: {image_path}\n")
        return 1
    items = extract_items(ocr.predict(str(image_path)), args.min_conf)
    print(json.dumps(items, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
