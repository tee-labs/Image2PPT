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

2. Use the path the user provided directly. `convert.py` accepts either a
   single image or a directory. Supported extensions are PNG, JPG/JPEG,
   WebP, BMP, and TIF/TIFF.

3. Run the one-command pipeline:

   ```bash
   python scripts/convert.py --source <source-image-or-dir>
   ```

4. Report:

   - final PPTX: `output_project/<run>/slides.pptx`
   - QA report: `output_project/<run>/qa.json`
   - previews: `output_project/<run>/previews/`
   - any known fidelity issues

## Notes

- Do not commit `output_project/`, source-private slides, caches, or model
  weights.
- Prefer the existing scripts and workflow instead of inventing a separate
  conversion path.
- If `build_deck.py` reports uncertain OCR entries, ask whether the user
  wants a manual review pass before rebuilding.
