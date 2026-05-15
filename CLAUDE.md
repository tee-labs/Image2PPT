# Claude Code Guide

DeckWeaver converts slide images into editable PowerPoint decks. This file
is for Claude Code. The same workflow is also documented in `AGENTS.md`,
`SKILL.md`, and `README.md`.

## Common User Request

The user may say:

> Use this skill to convert one image, or all images in a folder, into an
> editable PPT.

Use the local scripts in this repository. The main output is:

```text
output_project/<run>/slides.pptx
```

## Workflow

Install dependencies on first use:

```bash
bash scripts/bootstrap.sh
```

For a folder of slide images:

```bash
RUN="output_project/<name>_$(date +%Y%m%d_%H%M%S)"
SRC="<source-dir>"

python scripts/ocr/prepare_ocr.py --source-dir "$SRC" --work-dir "$RUN"
python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"
python scripts/build_deck.py --source-dir "$SRC" --work-dir "$RUN"
```

For a single image, place it in a temporary folder as `page_01.<ext>` and
run the same commands.

Supported input formats: PNG, JPG/JPEG, WebP, BMP, TIF/TIFF.

## Report Back

Always tell the user:

- where the PPTX was generated;
- whether QA passed;
- where preview images are;
- whether any OCR/manual review is recommended.

Do not commit generated runs, private source images, caches, model files,
or debug output unless the user explicitly asks.
