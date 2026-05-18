#!/usr/bin/env python
"""End-to-end deck builder: run_pipeline -> combine -> build -> QA.

Single-command orchestrator that runs the five-step build/QA sequence
in one go:

    1. run_pipeline.py    erase + inventory + layout (every page)
    2. combine_layouts.py per-page layouts -> deck-wide combined.layout.json
    3. build_pptx_from_layout.py   layout JSON -> slides.pptx
    4. inspect_pptx.py    package + media QA report -> qa.json
    5. render_preview.py  PPTX -> PDF -> per-page PNG previews

Run AFTER the OCR pre-pass (prepare_ocr.py + ocr_review_apply.py).
The work directory's ocr/ subfolder must already contain
`page_NN.ocr.json` files with the agent's review applied.

Usage:
    python scripts/build_deck.py \\
        --source-dir <slides_image_dir> \\
        --work-dir   output_project/<name>_<YYYYMMDD_HHMMSS>/

The orchestrator imports `run_pipeline` in-process so the per-page
loop reuses one Python session. The other four stages are simple
shell-outs because each one is already a thin standalone tool and
the subprocess overhead is small compared to LibreOffice startup
(render_preview alone is ~10 s).

Flags forwarded to run_pipeline:
    --pages 1,4,8           run only specified pages
    --detect-tables         enable SLANet table verification
    --table-score-threshold 0.85
    --icon-review           emit icon-review packet
    --icon-decisions        apply previously-recorded icon decisions

Stage skip flags (when iterating):
    --skip-pipeline         skip stage 1 (already-built layouts)
    --skip-render           skip stage 5 (no LibreOffice / preview need)
    --skip-calibration      skip preview-based text size/position calibration
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Per-page scripts live in scripts/page/; tables/ is consulted by
# run_pipeline. We're at scripts/ root.
SCRIPTS_ROOT = Path(__file__).resolve().parent      # .../scripts
sys.path.insert(0, str(SCRIPTS_ROOT / "page"))
sys.path.insert(0, str(SCRIPTS_ROOT / "tables"))

import run_pipeline as rp  # noqa: E402
from image_sources import supported_image_formats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source-dir", required=True,
                   help=f"Directory of page_NN slide images "
                        f"({supported_image_formats()}).")
    p.add_argument("--work-dir", required=True,
                   help="Run dir; expects ocr/page_NN.ocr.json present.")
    p.add_argument("--pages",
                   help="Comma-separated page numbers; default = all "
                        "ocr/page_NN.ocr.json present.")
    p.add_argument("--detect-tables", dest="detect_tables",
                   action="store_true",
                   help="Forward --detect-tables to run_pipeline.")
    p.add_argument("--no-detect-tables", dest="detect_tables",
                   action="store_false",
                   help="Forward --no-detect-tables to run_pipeline.")
    p.set_defaults(detect_tables=False)
    p.add_argument("--table-score-threshold", type=float, default=0.85)
    p.add_argument("--icon-review", action="store_true")
    p.add_argument("--icon-decisions", action="store_true")
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip stage 1 (use existing per-page layouts).")
    p.add_argument("--skip-render", action="store_true",
                   help="Skip stage 5 (no preview PNGs produced).")
    p.add_argument("--calibrate-positions", dest="calibrate_text",
                   action="store_true", default=None,
                   help="Force preview-based closed-loop text "
                        "calibration, even with --skip-render.")
    p.add_argument("--skip-calibration", dest="calibrate_text",
                   action="store_false",
                   help="Skip preview-based text size/position calibration.")
    p.add_argument("--font-calibration-iterations", type=int, default=1,
                   help="Text font-size calibration iterations (default: 1).")
    p.add_argument("--calibration-iterations", type=int, default=2,
                   help="Text position calibration iterations (default: 2).")
    p.add_argument("--calibration-max-shift", type=float, default=30.0,
                   help="Max source-pixel shift per text box per calibration "
                        "iteration (default: 30).")
    return p.parse_args()


def banner(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    args = parse_args()
    work = Path(args.work_dir)
    src = Path(args.source_dir)
    layouts_dir = work / "layouts"
    combined_path = layouts_dir / "combined.layout.json"
    pptx_path = work / "slides.pptx"
    draft_pptx_path = work / "slides.draft.pptx"
    qa_path = work / "qa.json"
    previews_dir = work / "previews"
    work.mkdir(parents=True, exist_ok=True)
    layouts_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ---- Stage 1: run_pipeline ----
    if not args.skip_pipeline:
        banner("1/5  run_pipeline (erase + inventory + layout)")
        ts = time.time()
        nums = (
            [n.strip().zfill(2) for n in args.pages.split(",")]
            if args.pages else rp.page_numbers(args)
        )
        if not nums:
            sys.stderr.write(
                f"ERROR: no page_NN.ocr.json under {work / 'ocr'}\n"
            )
            return 1
        for n in nums:
            t = time.time()
            try:
                r = rp.process_page(
                    n, src, work,
                    detect_tables_flag=args.detect_tables,
                    table_score_threshold=args.table_score_threshold,
                    icon_review_dump=args.icon_review,
                    icon_decisions=args.icon_decisions,
                )
            except Exception as exc:
                sys.stderr.write(f"page {n}: ERROR {exc}\n")
                raise
            tables_note = f" tables={r['tables']}" if r.get("tables") else ""
            print(f"page {n}: text={r['text']:>3} image={r['image']:>3}"
                  f"{tables_note} ({time.time() - t:.1f}s)", flush=True)
        print(f"  stage 1 done in {time.time() - ts:.1f}s", flush=True)
    else:
        banner("1/5  run_pipeline SKIPPED")

    # ---- Stage 2: combine_layouts ----
    banner("2/5  combine_layouts")
    ts = time.time()
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "deck" / "combine_layouts.py"),
         "--layouts", str(layouts_dir),
         "--out", str(combined_path)],
        check=True, capture_output=True, text=True,
    )
    print(r.stdout.strip())
    print(f"  stage 2 done in {time.time() - ts:.1f}s", flush=True)

    # ---- Stage 2b: structural text-slot classes ----
    banner("2b/5  classify_text_slots")
    ts = time.time()
    slot_report = work / "debug" / "text_slot_classes.json"
    r = subprocess.run(
        [sys.executable,
         str(SCRIPTS_ROOT / "deck" / "classify_text_slots.py"),
         "--layout", str(combined_path),
         "--out", str(slot_report),
         "--apply",
         "--min-group-size", "2",
         "--min-apply-size", "3"],
        check=True, capture_output=True, text=True,
    )
    if r.stdout.strip():
        print(r.stdout.strip())
    print(f"  -> {slot_report}")
    print(f"  stage 2b done in {time.time() - ts:.1f}s", flush=True)

    should_calibrate = (
        (not args.skip_render)
        if args.calibrate_text is None else args.calibrate_text
    )

    # ---- Stage 3: build_pptx_from_layout ----
    banner("3/5  build_pptx_from_layout")
    ts = time.time()
    stage3_pptx_path = draft_pptx_path if should_calibrate else pptx_path
    r = subprocess.run(
        [sys.executable,
         str(SCRIPTS_ROOT / "deck" / "build_pptx_from_layout.py"),
         "--layout", str(combined_path),
         "--assets-root", str(work),
         "--out", str(stage3_pptx_path)],
        check=True, capture_output=True, text=True,
    )
    if r.stdout.strip():
        print(r.stdout.strip())
    print(f"  -> {stage3_pptx_path}")
    print(f"  stage 3 done in {time.time() - ts:.1f}s", flush=True)

    # ---- Stage 3b/3c: closed-loop text calibration ----
    if should_calibrate:
        banner("3b/5  calibrate_text_sizes")
        ts = time.time()
        r = subprocess.run(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "calibrate_text_sizes.py"),
             "--layout", str(combined_path),
             "--source-dir", str(src),
             "--work-dir", str(work),
             "--assets-root", str(work),
             "--iterations", str(args.font_calibration_iterations)],
            check=True, capture_output=True, text=True,
        )
        if r.stdout.strip():
            print(r.stdout.strip())
        slot_report = work / "debug" / "text_slot_classes.after_size.json"
        r = subprocess.run(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "classify_text_slots.py"),
             "--layout", str(combined_path),
             "--out", str(slot_report),
             "--apply",
             "--min-group-size", "2",
             "--min-apply-size", "3"],
            check=True, capture_output=True, text=True,
        )
        if r.stdout.strip():
            print(r.stdout.strip())
        r = subprocess.run(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "build_pptx_from_layout.py"),
             "--layout", str(combined_path),
             "--assets-root", str(work),
             "--out", str(draft_pptx_path)],
            check=True, capture_output=True, text=True,
        )
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {draft_pptx_path}")
        print(f"  stage 3b done in {time.time() - ts:.1f}s", flush=True)

        banner("3c/5  calibrate_text_positions")
        ts = time.time()
        r = subprocess.run(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "calibrate_text_positions.py"),
             "--layout", str(combined_path),
             "--source-dir", str(src),
             "--work-dir", str(work),
             "--assets-root", str(work),
             "--iterations", str(args.calibration_iterations),
             "--max-shift", str(args.calibration_max_shift)],
            check=True, capture_output=True, text=True,
        )
        if r.stdout.strip():
            print(r.stdout.strip())
        slot_report = work / "debug" / "text_slot_classes.after_position.json"
        r = subprocess.run(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "classify_text_slots.py"),
             "--layout", str(combined_path),
             "--out", str(slot_report),
             "--apply",
             "--min-group-size", "2",
             "--min-apply-size", "3"],
            check=True, capture_output=True, text=True,
        )
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {slot_report}")
        r = subprocess.run(
            [sys.executable,
             str(SCRIPTS_ROOT / "deck" / "build_pptx_from_layout.py"),
             "--layout", str(combined_path),
             "--assets-root", str(work),
             "--out", str(pptx_path)],
            check=True, capture_output=True, text=True,
        )
        if r.stdout.strip():
            print(r.stdout.strip())
        print(f"  -> {pptx_path}")
        print(f"  stage 3c done in {time.time() - ts:.1f}s", flush=True)
    else:
        banner("3b/5  calibrate_text_sizes SKIPPED")
        banner("3c/5  calibrate_text_positions SKIPPED")

    # ---- Stage 4: inspect_pptx ----
    banner("4/5  inspect_pptx")
    ts = time.time()
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_ROOT / "verify" / "inspect_pptx.py"),
         "--pptx", str(pptx_path),
         "--report", str(qa_path)],
        check=True, capture_output=True, text=True,
    )
    print(r.stdout.strip())
    print(f"  stage 4 done in {time.time() - ts:.1f}s", flush=True)
    try:
        report = json.loads(qa_path.read_text(encoding="utf-8"))
        failures = report.get("failures", [])
        if failures:
            sys.stderr.write(
                f"WARNING: {len(failures)} QA failures — see {qa_path}\n"
            )
    except (OSError, json.JSONDecodeError):
        pass

    # ---- Stage 5: render_preview ----
    if not args.skip_render:
        banner("5/5  render_preview (PPTX -> PDF -> PNG)")
        ts = time.time()
        previews_dir.mkdir(parents=True, exist_ok=True)
        for stale in previews_dir.glob("page-*.png"):
            stale.unlink()
        try:
            # render_preview writes <out-dir>/previews/page-NN.png,
            # so we point --out-dir at <work> to land at <work>/previews/.
            r = subprocess.run(
                [sys.executable, str(SCRIPTS_ROOT / "verify" / "render_preview.py"),
                 "--pptx", str(pptx_path),
                 "--out-dir", str(work)],
                check=True, capture_output=True, text=True,
            )
            if r.stdout.strip():
                print(r.stdout.strip())
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(
                f"WARNING: render_preview failed: {exc.stderr or exc}\n"
                "         (LibreOffice / pdftoppm may not be installed.)\n"
            )
        print(f"  stage 5 done in {time.time() - ts:.1f}s", flush=True)
    else:
        banner("5/5  render_preview SKIPPED")

    print(f"\n=== build_deck total: {time.time() - t_total:.1f}s ===")
    print(f"  PPTX:     {pptx_path}")
    print(f"  QA:       {qa_path}")
    if not args.skip_render:
        print(f"  Previews: {previews_dir}/")

    # ---- Post-build: OCR review opportunity summary ----
    # Lists pages where 3-engine consensus left yellow/red entries the
    # agent might want to verify. The deck already uses the consensus
    # picks, so this is OPTIONAL — surface it so the calling agent can
    # ask the user whether to do a manual review pass.
    _report_review_pending(work)

    return 0


def _report_review_pending(work: Path) -> None:
    """Scan ocr/page_*.ocr_review.json and print per-page yellow/red counts.

    Skips silently if no review JSONs exist (e.g. legacy workdir or the
    user ran build_deck.py without prepare_ocr.py first).
    """
    ocr_dir = work / "ocr"
    review_paths = sorted(ocr_dir.glob("page_*.ocr_review.json"))
    if not review_paths:
        return

    rows = []
    total_y = total_r = 0
    for rp in review_paths:
        try:
            d = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Count YELLOW + RED entries that aren't auto-cleared. These
        # have a pre-filled corrected_text but are visually flagged on
        # the annotated page for the agent to consider.
        y = r = 0
        for e in d.get("entries", []):
            if (e.get("notes") or "").startswith("auto-cleared"):
                continue
            if e.get("tier") == "yellow":
                y += 1
            elif e.get("tier") == "red":
                r += 1
        if y or r:
            num = rp.stem.replace(".ocr_review", "").split("_")[1]
            rows.append((num, y, r))
            total_y += y
            total_r += r

    if not rows:
        return

    print("\n=== Optional: OCR review ===")
    print(f"  {total_y + total_r} entries across {len(rows)} pages have "
          f"3-engine consensus pre-fills the PPTX is using.")
    print(f"  ({total_y} 🟡 weak consensus — glance to verify; "
          f"{total_r} 🔴 no consensus — review carefully)")
    print()
    print("  Pages with pending review entries:")
    for num, y, r in rows:
        ann = work / "ocr" / f"page_{num}.ocr_review.annotated.png"
        print(f"    page {num}: {y} 🟡 + {r} 🔴   {ann}")
    print()
    print("  ASK THE USER whether to manually review these. If yes:")
    print("    1. open each ocr_review.annotated.png above")
    print("    2. for any wrong pre-fill, edit `corrected_text` in the")
    print("       matching ocr_review.json")
    print("    3. rerun (FULL rebuild — OCR text propagates through")
    print("       erase + layout → PPTX):")
    print(f"         scripts/ocr/ocr_review_apply.py --work-dir {work}")
    print(f"         scripts/build_deck.py --source-dir <src> "
          f"--work-dir {work}")


if __name__ == "__main__":
    raise SystemExit(main())
