#!/usr/bin/env python3
"""Ingest a PDF into a run dir as `page_NN.png` + native OCR JSON.

PDF text is extracted directly from the PDF text layer with PyMuPDF, so
the resulting `page_NN.ocr.json` carries exact Unicode strings and
per-character bboxes (confidence = 1.0). The rendered PNG is still
written so the rest of the pipeline (erase, image inventory, layout,
calibration) keeps its visual reference unchanged.

Outputs under --work-dir:

    source/page_NN.png            rendered page at --dpi
    ocr/page_NN.ocr.json          native text + char boxes
    ocr/page_NN.ocr_review.json   empty review packet (so re-running
                                  ocr_review_apply manually is a no-op)

Usage:
    python scripts/ocr/pdf_ingest.py --pdf deck.pdf --work-dir output/<run>/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pdf_ingest requires PyMuPDF. Install with: pip install pymupdf"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", required=True, help="Source PDF path.")
    p.add_argument("--work-dir", required=True, help="Run directory.")
    p.add_argument("--dpi", type=int, default=300,
                   help="Render DPI for page PNGs (default 300).")
    p.add_argument("--pages", default=None,
                   help="Comma-separated 1-based page numbers to ingest.")
    return p.parse_args()


def parse_pages(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(tok)
        if n < 1 or n > total:
            raise SystemExit(f"--pages: {n} out of range 1..{total}")
        out.append(n)
    return out


def _line_entry(line: dict, scale: float) -> dict | None:
    """Build one OCR entry from a PyMuPDF rawdict line."""
    chars: list[str] = []
    boxes: list[list[int]] = []
    for span in line.get("spans", []):
        for ch in span.get("chars", []):
            c = ch.get("c", "")
            if not c or c.isspace():
                # Whitespace chars carry no bbox useful for layout; keep
                # them inside the joined text but skip per-char boxes so
                # downstream word-aligned reflow stays clean.
                if c:
                    chars.append(c)
                    boxes.append(None)  # placeholder, removed below
                continue
            bx = ch.get("bbox")
            if not bx or len(bx) != 4:
                continue
            chars.append(c)
            boxes.append([
                int(round(bx[0] * scale)),
                int(round(bx[1] * scale)),
                int(round(bx[2] * scale)),
                int(round(bx[3] * scale)),
            ])

    # Drop trailing whitespace placeholders before computing the line bbox.
    while chars and chars[-1].isspace():
        chars.pop()
        boxes.pop()
    while chars and chars[0].isspace():
        chars.pop(0)
        boxes.pop(0)
    if not chars:
        return None

    # If any char has no bbox (pure whitespace inside the line), drop the
    # placeholder slots from word_boxes so words[i]/word_boxes[i] stay
    # aligned: collapse whitespace into the preceding glyph by widening
    # its bbox to the right slightly.
    text_chars: list[str] = []
    text_boxes: list[list[int]] = []
    for c, b in zip(chars, boxes):
        if b is None:
            if text_boxes:
                # Extend last bbox right edge by a small whitespace stub.
                # Half a char width is a safe default visually.
                w = text_boxes[-1][2] - text_boxes[-1][0]
                text_boxes[-1][2] += max(w // 2, 1)
                text_chars[-1] += c
            else:
                continue
        else:
            text_chars.append(c)
            text_boxes.append(b)

    if not text_chars:
        return None

    x1 = min(b[0] for b in text_boxes)
    y1 = min(b[1] for b in text_boxes)
    x2 = max(b[2] for b in text_boxes)
    y2 = max(b[3] for b in text_boxes)

    return {
        "text": "".join(text_chars),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "confidence": 1.0,
        "words": text_chars,
        "word_boxes": text_boxes,
    }


def extract_text_entries(page: "fitz.Page", scale: float) -> list[dict]:
    raw = page.get_text("rawdict")
    entries: list[dict] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            entry = _line_entry(line, scale)
            if entry is not None:
                entries.append(entry)
    # Stable top-to-bottom, left-to-right ordering — matches the order
    # PaddleOCR produces and what downstream layout expects.
    entries.sort(key=lambda e: (e["y1"], e["x1"]))
    return entries


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Not a .pdf file: {pdf_path}")

    work = Path(args.work_dir).expanduser()
    src_dir = work / "source"
    ocr_dir = work / "ocr"
    src_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        page_nums = parse_pages(args.pages, doc.page_count)
        scale = args.dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        print(f"[pdf_ingest] {pdf_path.name} → {len(page_nums)} pages "
              f"@ {args.dpi} dpi", flush=True)

        for i, n in enumerate(page_nums, 1):
            page = doc.load_page(n - 1)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png_path = src_dir / f"page_{i:02d}.png"
            pix.save(str(png_path))

            entries = extract_text_entries(page, scale)
            ocr_path = ocr_dir / f"page_{i:02d}.ocr.json"
            review_path = ocr_dir / f"page_{i:02d}.ocr_review.json"
            ocr_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Empty review packet — ocr_review_apply.py is a no-op if rerun.
            review_path.write_text(
                json.dumps(
                    {"image": png_path.name, "entries": [],
                     "tier_counts": {"green": 0, "yellow": 0, "red": 0}},
                    ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  page {i:02d}: {pix.width}x{pix.height}px, "
                  f"{len(entries)} text lines", flush=True)
    finally:
        doc.close()

    print(f"[pdf_ingest] done. source/ and ocr/ ready under {work}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
