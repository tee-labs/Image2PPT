<p align="center">
  <img src="assets/logo.png" alt="DeckWeaver logo" width="420">
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

### Option 1: Use It As A Skill / Agent Tool

Use this path if you work with Codex, Claude Code, or another local
coding agent.

Clone the project:

```bash
git clone https://github.com/GuopengLin/Image2PPT.git
cd Image2PPT
```

Or clone it directly into a Codex skill directory:

```bash
git clone https://github.com/GuopengLin/Image2PPT.git ~/.codex/skills/deckweaver
```

Then open the project with your agent and ask it to convert one image or
all images in a folder, for example:

```text
Use this skill to convert all images under slides/ into an editable PPT.
```

For first-time setup, ask the agent to run:

```bash
bash scripts/bootstrap.sh
```

Then ask it to run:

```bash
python scripts/convert.py --source slides
```

Codex, Claude Code, and general agents can use `AGENTS.md` and
`SKILL.md`.

### Option 2: Use It As A Standalone CLI Tool

Clone and install dependencies:

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

Run the one-command pipeline:

```bash
python scripts/convert.py --source slides
```

For a single image:

```bash
python scripts/convert.py --source /path/to/page_01.png
```

For debugging, the one-command flow can still be split into the three
lower-level steps:

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
- `SKILL.md`: skill workflow used by compatible agents, including
  Claude Code when it is configured to read project skill instructions.

## Contact

Commercial licensing, customization, or feedback: 1015277323@qq.com

## License

Free for personal use. Commercial use, redistribution, SaaS use, or production integration requires a paid commercial license. See [LICENSE](LICENSE).
