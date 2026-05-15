---
name: ppt-image-to-editable-ppt
description: Convert PPT slide screenshots or exported slide images into editable PowerPoint decks. Use when Codex needs to extract image/icon/material assets from one or many slide images as separate PNGs, rebuild the slides with editable text boxes, native shapes, and movable picture objects, batch-process multiple page images, and merge the reconstructed pages into a complete .pptx file.
---

# PPT Image To Editable PPT

## Overview

Turn slide images into a reconstructed, editable PowerPoint deck. Text
stays as real editable PowerPoint text. Icons, logos, photos and
decorative bands come out as separate movable PNG picture objects.
Cards, frames, circles and rules are rebuilt as native shapes/lines
where the source geometry is simple enough.

This skill is for slide screenshots or exported slide images. If a
native `.pptx` already exists, edit that directly.

Supported input formats use `page_NN.<ext>` naming and include PNG,
JPG/JPEG, WebP, BMP, and TIF/TIFF. Extracted assets and previews are
written as PNG.

## Dependencies

Run the bootstrap script from the repository root:

```bash
bash scripts/bootstrap.sh
```

It installs the Python packages in `requirements.txt`, local OCR tools,
LibreOffice, Poppler, and model caches where the platform supports
automatic installation. On managed systems, install these manually and
use `bash scripts/bootstrap.sh --no-system`.

## Run Directory

Every job goes under:

```text
output_project/<name>_<YYYYMMDD_HHMMSS>/
├── slides.pptx
├── qa.json
├── previews/
├── ocr/
├── inventory/
├── manifests/
├── layouts/
├── assets/
└── debug/
```

## Workflow

Prepare source images:

```text
slides/
├── page_01.png
├── page_02.jpg
└── page_03.webp
```

Run:

```bash
RUN="output_project/demo_$(date +%Y%m%d_%H%M%S)"
SRC="slides"

python scripts/ocr/prepare_ocr.py \
  --source-dir "$SRC" \
  --work-dir "$RUN"

python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"

python scripts/build_deck.py \
  --source-dir "$SRC" \
  --work-dir "$RUN"
```

`prepare_ocr.py` loads PaddleOCR once, runs OCR across all pages, builds
review packets for uncertain entries, and pre-fills `corrected_text`
with local consensus picks. `ocr_review_apply.py` merges those picks into
the OCR JSON files. `build_deck.py` runs erase, inventory extraction,
layout generation, deck assembly, QA inspection, and preview rendering.

## Optional Review

After `build_deck.py`, check whether it reports OCR entries that may
benefit from manual review. If the user wants the extra pass:

1. Open `ocr/page_NN.ocr_review.annotated.png`.
2. Edit `ocr/page_NN.ocr_review.json`.
3. Set `corrected_text` to the desired text, or to `""` when the OCR
   detection is not real editable text.
4. Rerun:

```bash
python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"
python scripts/build_deck.py --source-dir "$SRC" --work-dir "$RUN"
```

## Useful Flags

- `--pages 1,4,8`: process only selected pages.
- `--skip-render`: skip LibreOffice preview generation.
- `--detect-tables`: enable optional native table reconstruction.
- `--icon-review`: emit icon-vs-text review packets.
- `--icon-decisions`: apply filled icon review decisions on a rerun.

## Validation Standard

A job is done when:

- meaningful visual assets are present as PNG picture objects or native
  shapes;
- ordinary text is editable in the PPTX;
- extracted PNG assets are independent objects, not a flattened full-page
  background;
- `inspect_pptx.py` reports no failures;
- previews have been compared with the source images, or the final
  response states why preview rendering was skipped or unavailable.

Final responses should include the final `.pptx` path, the preview
directory, the QA report, and any known fidelity differences.
