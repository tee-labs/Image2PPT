<p align="center">
  <img src="assets/logo.png" alt="DeckWeaver logo" width="220">
</p>

# DeckWeaver

[中文](README.md) | English

DeckWeaver reconstructs slide screenshots, exported slide images, or visual slide drafts into editable PowerPoint decks. It keeps ordinary text as real PPT text boxes, extracts icons/logos/photos as independent picture objects, and rebuilds simple cards, lines, and rounded rectangles as native shapes where possible.

The repository name remains Image2PPT so the project is easy to find and understand at a glance.

Use it when you only have images. If you already have the native `.pptx`, edit that file directly.

## Highlights

- Nearly all text and icons remain editable, movable, and replaceable in PowerPoint.
- Very low token usage: the main pipeline uses local OCR and image processing instead of repeatedly sending full slides to a cloud multimodal model.
- Fast batch generation with a warm OCR process and automated page stages.
- No additional cloud API required.
- Supports common bitmap inputs: PNG, JPG/JPEG, WebP, BMP, TIF/TIFF.

## Quick Start

```bash
git clone https://github.com/GuopengLin/Image2PPT.git
cd Image2PPT
bash scripts/bootstrap.sh
```

Prepare source images named by page number:

```text
slides/
├── page_01.png
├── page_02.jpg
├── page_03.webp
└── page_04.tiff
```

Run the pipeline:

```bash
RUN="output_project/demo_$(date +%Y%m%d_%H%M%S)"
SRC="slides"

python scripts/ocr/prepare_ocr.py --source-dir "$SRC" --work-dir "$RUN"
python scripts/ocr/ocr_review_apply.py --work-dir "$RUN"
python scripts/build_deck.py --source-dir "$SRC" --work-dir "$RUN"
```

The final deck is written to `output_project/<run>/slides.pptx`. QA output, preview renders, OCR data, extracted assets, and layout JSON files are stored in the same run directory.

## Agent Support

- `AGENTS.md`: guide for Codex and general coding agents.
- `CLAUDE.md`: guide for Claude Code.
- `agents/codex.yaml`: optional Codex/OpenAI-style UI metadata.

## Contact

Commercial licensing, customization, or feedback: 1015277323@qq.com

## License

Free for personal use. Commercial use, redistribution, SaaS use, or production integration requires a paid commercial license. See [LICENSE](LICENSE).
