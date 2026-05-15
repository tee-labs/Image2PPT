# Reconstruction SOP

## 1. Prepare Source Images

Use one image per slide, named by page number:

```text
slides/
├── page_01.png
├── page_02.jpg
└── page_03.webp
```

Supported formats are PNG, JPG/JPEG, WebP, BMP, and TIF/TIFF. Keep only
one source image per page number.

## 2. Run OCR

```bash
RUN="output_project/demo_$(date +%Y%m%d_%H%M%S)"
SRC="slides"

python scripts/ocr/prepare_ocr.py \
  --source-dir "$SRC" \
  --work-dir "$RUN"
```

This creates OCR JSON files and, when needed, annotated review images.

## 3. Apply OCR Decisions

```bash
python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"
```

The first pass can run without manual edits because uncertain entries are
pre-filled with local consensus suggestions.

## 4. Build the Deck

```bash
python scripts/build_deck.py \
  --source-dir "$SRC" \
  --work-dir "$RUN"
```

The final PPTX is written to `slides.pptx` in the run directory.

## 5. Optional Manual Review

If the build reports uncertain OCR entries:

1. Open `ocr/page_NN.ocr_review.annotated.png`.
2. Edit `ocr/page_NN.ocr_review.json`.
3. Set `corrected_text` to the intended text, or `""` for non-text
   decorative detections.
4. Rerun `ocr_review_apply.py` and `build_deck.py`.

## 6. QA Checklist

- Ordinary text is editable.
- Visual assets are independent picture objects or intentional native
  shapes.
- No important element is missing.
- `qa.json` has no failures.
- `previews/page-NN.png` images have been compared with the source.
- Any remaining differences are documented for the user.
