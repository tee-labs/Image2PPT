# Auto-Extract Pipeline

This document summarizes the local OCR + image-processing path used by
Image2PPT. It intentionally avoids source-deck-specific examples so the
repository can be published cleanly.

## Input

Source images must be named `page_NN.<ext>`.

Supported extensions:

- `.png`
- `.jpg` / `.jpeg`
- `.webp`
- `.bmp`
- `.tif` / `.tiff`

Each page number should appear once. If both `page_01.png` and
`page_01.jpg` exist, the pipeline stops and asks the user to keep one.

## Stages

1. `prepare_ocr.py`

   Runs local OCR on every source page, cross-checks uncertain text with
   additional local engines, writes `ocr/page_NN.ocr.json`, and creates
   optional annotated review images.

2. `ocr_review_apply.py`

   Applies the chosen `corrected_text` values back into the OCR JSON
   files. The first run can be fully unattended because review entries
   are pre-filled.

3. `run_pipeline.py`

   For each page, erases editable text from the source image, detects
   remaining visual components, extracts assets, and writes a page layout
   JSON.

4. `combine_layouts.py`

   Combines per-page layout JSON files into one deck-level layout.

5. `build_pptx_from_layout.py`

   Writes the editable `.pptx`, using native text boxes and shapes where
   possible and independent picture objects for extracted visuals.

6. `inspect_pptx.py` and `render_preview.py`

   Produce QA metadata and preview images for visual comparison.

## Outputs

```text
output/<run>/
├── slides.pptx
├── qa.json
├── previews/page-NN.png
├── ocr/page_NN.ocr.json
├── ocr/page_NN.ocr_review.json
├── inventory/page_NN.inventory.json
├── manifests/page_NN.assets.json
├── layouts/page_NN.layout.json
├── layouts/combined.layout.json
├── assets/page_NN/*.png
└── debug/page_NN_*.png
```

## Extraction Rules

The pipeline attempts to keep ordinary text editable and visual material
movable. As a rule of thumb:

- text, captions, labels, bullets, and numbers become PPT text boxes;
- icons, logos, photos, decorative bands, and complex artwork become
  independent PNG objects;
- simple rectangles, rounded rectangles, lines, and circles may become
  native PowerPoint shapes;
- complex charts are usually preserved as picture objects unless a
  separate reconstruction path is added.

OCR and icon classification are heuristic. When fidelity matters, inspect
the preview images and use the optional OCR/icon review passes.
