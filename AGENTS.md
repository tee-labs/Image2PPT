# AGENTS.md

Workspace guide for ZCode agents working in this repo (DeckWeaver / Image2PPT).

## What this project is

DeckWeaver turns slide screenshots, exported slide images, or PDFs into
**editable** PowerPoint (`.pptx`) files. Text becomes real editable text
boxes; icons/logos/photos become separate movable PNG picture objects;
simple geometry becomes native shapes/lines. The heavy lifting is local
OCR + image algorithms — no cloud multimodal API is required.

It ships as three independent entry points that share the same pipeline:

1. **CLI** — `python scripts/convert.py --source <img|dir|pdf>`
2. **Agent skill** — `SKILL.md` describes the workflow an agent follows.
3. **Web service** — FastAPI backend + React/Vite frontend under `web/`.

## Major directories

```
scripts/        CLI pipeline (the core of the project)
  convert.py        one-command entry; shells out to the 3 steps below
  ocr/              OCR, cross-verify, review-apply, PDF text-layer ingest
  page/             per-page: erase text, detect elements, build layout
  deck/             combine layouts -> build PPTX -> calibrate -> QA
  verify/           PPTX inspection + LibreOffice preview rendering
  tables/           optional native table reconstruction
  icon/             icon detection / inpaint
  optional/, shared/
web/            additive web layer (does NOT affect CLI behavior)
  backend/app/      FastAPI app; spawns scripts/convert.py as subprocess
  frontend/         React + Vite SPA (TypeScript)
references/     layout JSON format, reconstruction SOP, auto-extract notes
tests/          metric-based regression suite (fixtures + runner)
models/         shipped model weights (font_classifier.onnx is committed)
assets/         logo / example images
```

All per-run output lands under `output/<name>_<YYYYMMDD>/` (gitignored).
Web runtime state lands under `web/data/` (gitignored).

## Build & run commands

Setup (idempotent — installs Python deps, system tools, model caches):

```bash
bash scripts/bootstrap.sh            # auto CPU/GPU
bash scripts/bootstrap.sh --no-system   # pip + warmup only
```

Run the pipeline:

```bash
python scripts/convert.py --source slides/                 # folder
python scripts/convert.py --source slides/ --pages 1,3,8   # subset
python scripts/convert.py --source page_01.png             # single image
python scripts/convert.py --source deck.pdf --pdf-dpi 300  # PDF
```

For debugging, the one-command flow splits into three steps
(see `references/reconstruction-sop.md`):

```bash
python scripts/ocr/prepare_ocr.py   --source-dir "$SRC" --work-dir "$RUN"
python scripts/ocr/ocr_review_apply.py        --work-dir "$RUN"
python scripts/build_deck.py         --source-dir "$SRC" --work-dir "$RUN"
```

Web layer (separate deps — CLI users never install fastapi/uvicorn):

```bash
bash web/start.sh        # dev: uvicorn :8000 + vite :5173
bash web/start-prod.sh   # prod: builds frontend, single uvicorn :8000
```

Frontend typecheck + build (from `web/frontend/`):

```bash
npm run build     # tsc -b && vite build
npm run dev
```

Regression tests (metric-based, NOT pass/fail — `tests/README.md`):

```bash
python tests/runner/runner.py                       # all fixtures vs baseline
python tests/runner/runner.py -k light_cover_hero   # single fixture
python tests/runner/runner.py --rerun-pipeline      # bypass cache
python tests/runner/runner.py --update-baseline     # record current scores
```

Unit/smoke tests (offline; pytest-style, also discovers the `unittest`
tests under `tests/`):

```bash
pip install -r requirements-dev.txt
pytest tests/test_convert_vlm_smoke.py   # VLM renderer contract (no API key)
pytest                                    # all unit tests
```

There is no Python lint/typecheck config in the repo; match existing style.

## Architecture boundaries (matter for edits)

- **The web layer is additive.** It must never be required by the CLI. Web
  deps live in `web/backend/requirements.txt`, separate from the root
  `requirements.txt`. The web backend spawns `scripts/convert.py` as a
  subprocess with `CWD = repo root` — do not move that script or change its
  CLI/stdout contract without updating `web/backend/app/runner.py`.
- **`build_deck.py` orchestrates five stages** in one Python session:
  `run_pipeline` (imported in-process) → `combine_layouts` →
  `build_pptx_from_layout` → `inspect_pptx` → `render_preview`. The first
  four are imported/shelled out as documented in its module docstring.
- **Layout JSON is the contract between page processing and PPTX building.**
  Read `references/layout-json.md` before touching anything in
  `scripts/page/layout/` or `scripts/deck/`. Coordinates are in
  source-image pixels and scaled to the slide size.
- **Web backend imports use package paths** (`web.backend.app.main:app`),
  so uvicorn must run from repo root. The `REPO_ROOT`/`WEB_ROOT`/`DATA_ROOT`
  anchors are defined in `web/backend/app/config.py`.
- **PDF input skips OCR**: `scripts/ocr/pdf_ingest.py` extracts the text
  layer directly (exact Unicode + per-glyph bbox, confidence 1.0) and
  rasterizes pages. `convert.py` short-circuits straight to `build_deck.py`
  in PDF mode.

## Conventions

- Page source images use `page_NN.<ext>` naming (PNG/JPG/WebP/BMP/TIFF).
  Non-conforming inputs are auto-copied into `<work>/source/` and renumbered.
- One source image per page number; one page size per PPTX.
- Run dirs: `output/<source-stem-or-dir-name>_<YYYYMMDD>/` with subfolders
  `previews/`, `ocr/`, `layouts/`, `assets/`, `debug/`, plus `slides.pptx`
  and `qa.json`.
- A job is complete when: assets are independent objects (not a flattened
  background), text is editable, `inspect_pptx.py` reports no failures, and
  previews have been compared to source. See `SKILL.md` "Validation Standard".

## Known gotchas

- **Paddle stack is version-pinned for a reason.** The Dockerfile pins
  `paddleocr==3.7.0`, `paddlex[ocr]==3.7.2`, `paddlepaddle==3.2.2`
  (PP-OCRv6 era). The OCR scripts also pass
  `text_detection_model_name="PP-OCRv6_medium_det"` /
  `text_recognition_model_name="PP-OCRv6_medium_rec"` explicitly so the
  loaded model does not depend on the package default. Two independent
  version constraints are in play: (a) `paddleocr>=3.7.0` is the first
  release with the PP-OCRv6 model registry; (b) `paddlepaddle==3.2.2` is
  pinned back because 3.3+ hits a PIR/oneDNN conflict
  (`(Unimplemented) ConvertPirAttribute2RuntimeAttribute ...`,
  PaddlePaddle/Paddle#77340) on the CPU path — PP-OCRv6 officially
  supports paddlepaddle 3.1+, so 3.2.2 satisfies the floor while dodging
  the crash. If you bump, bump all three together and re-run the
  regression suite. If you must move to paddlepaddle 3.3+, add
  `enable_mkldnn=False` to the `PaddleOCR(...)` calls as a workaround
  (PaddleOCR#18162) at the cost of CPU oneDNN acceleration.
- **Docker requires `DECKWEAVER_SUBPROCESS_MEMORY_MB=0`.** The default 6 GB
  `RLIMIT_AS` is too small — PaddleOCR mmaps model weights + large tensors
  and exceeds it, raising `std::bad_alloc` (surfaced as OCR 0 detections /
  `convert.py exited with 1`). Memory caps belong on `docker run --memory`.
- **Docker has no default secrets.** `web/backend/app/main.py::_check_secrets`
  refuses to boot with weak `DECKWEAVER_ADMIN_PASSWORD` / `DECKWEAVER_JWT_SECRET`
  when `DECKWEAVER_REQUIRE_SECURE_SECRETS=true` (default). Override both via
  `-e`, or set `REQUIRE_SECURE_SECRETS=false` only for trusted single-machine deploys.
- **easyocr/torch is NOT installed in Docker** (would add ~1 GB). Cross-verify
  (`DECKWEAVER_CROSS_VERIFY`) defaults to off; only enable where both
  EasyOCR and Tesseract are installed.
- **VLM profile is opt-in and conflicts with the project's no-cloud default.**
  `DECKWEAVER_USE_VLM=true` makes the web runner invoke
  `scripts/convert_vlm.py` instead of `scripts/convert.py`. That path
  bypasses local OCR entirely and calls an OpenAI-compatible
  `/v1/chat/completions` endpoint, so it requires `DECKWEAVER_LLM_BASE` +
  `DECKWEAVER_LLM_KEY` (forwarded by `sandbox.safe_env`) and the `httpx`
  dependency (see `requirements-vps.txt`). Trade-off: the VLM profile emits
  text + vector shapes only — it does NOT extract logos/photos as
  independent picture objects, so rebuilt decks are lower fidelity than
  the local-OCR default. `deploy/deckweaver-web.service` is a systemd unit
  for the VPS/VLM deployment.
- **CJK font metrics matter.** The Dockerfile installs WenQuanYi/Noto CJK
  fonts and a fontconfig alias mapping Microsoft YaHei / 微软雅黑 / PingFang SC
  to metric-compatible substitutes. Font calibration relies on these aliases.
- **`models/font_classifier.onnx` is committed** despite `*.onnx` being
  gitignored (see the `!` exception in `.gitignore`). Don't delete it — the
  pipeline needs it after clone.

## Read before changing sensitive areas

- `references/layout-json.md` — layout JSON contract (text/image/shape/line).
- `references/reconstruction-sop.md` — the 3-step build flow and QA checklist.
- `references/auto-extract.md` — asset extraction heuristics.
- `SKILL.md` — the agent-facing skill workflow and validation standard.
- `tests/README.md` — how regression scoring works before editing
  `tests/runner/` or adding fixtures.
- `web/README.md` — web API, ETA formula, auto-update, user management CLI
  (`web/backend/manage.py`) before editing the web layer.
