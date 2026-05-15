# Agent Guide

This repository is a local image-to-editable-PPT skill. Use it to turn one
slide image or a folder of `page_NN.<ext>` slide images into an editable
PowerPoint deck.

## What To Do

When the user asks to convert images into PPT:

1. If this is the first run on the machine, run:

   ```bash
   bash scripts/bootstrap.sh
   ```

2. If the user provides one image, create a temporary source directory and
   copy or symlink it as `page_01.<ext>`.

3. If the user provides a folder, make sure supported images are named
   `page_NN.<ext>`. Supported extensions are PNG, JPG/JPEG, WebP, BMP,
   and TIF/TIFF.

4. Run:

   ```bash
   RUN="output_project/<name>_$(date +%Y%m%d_%H%M%S)"
   SRC="<source-dir>"

   python scripts/ocr/prepare_ocr.py --source-dir "$SRC" --work-dir "$RUN"
   python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"
   python scripts/build_deck.py --source-dir "$SRC" --work-dir "$RUN"
   ```

5. Report:

   - final PPTX: `$RUN/slides.pptx`
   - QA report: `$RUN/qa.json`
   - previews: `$RUN/previews/`
   - any known fidelity issues

## Notes

- Do not commit `output_project/`, source-private slides, caches, or model
  weights.
- Prefer the existing scripts and workflow instead of inventing a separate
  conversion path.
- If `build_deck.py` reports uncertain OCR entries, ask whether the user
  wants a manual review pass before rebuilding.
