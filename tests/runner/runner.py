"""CLI entry: run the regression suite against current pipeline state.

Usage:
    python tests/runner/runner.py                   # all fixtures vs baseline
    python tests/runner/runner.py -k <name>         # one fixture
    python tests/runner/runner.py --rerun-pipeline  # bypass pipeline cache
    python tests/runner/runner.py --update-baseline # write current scores as new baseline

The runner does NOT use pytest — keeping it standalone makes it usable
both as a developer tool and from CI without pytest discovery overhead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.runner.baseline import Comparison, Status, compare, write_baseline
from tests.runner.metrics import (
    FixtureScores, load_source_text_reference, score_fixture,
)
from tests.runner.pipeline import PipelineError, run_pipeline
from tests.runner.report import render_report
from tests.tools.fixture_specs import FIXTURE_SPECS, by_name

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-k", "--only", action="append", default=[],
                   help="Run only this fixture (repeatable).")
    p.add_argument("--rerun-pipeline", action="store_true",
                   help="Force re-run of convert.py (bypass cache).")
    p.add_argument("--update-baseline", action="store_true",
                   help="Overwrite baseline.json + baseline_reconstructed.png "
                        "with current scores.")
    p.add_argument("--no-report", action="store_true",
                   help="Skip HTML report (still prints summary).")
    return p.parse_args()


def select_fixtures(only: list[str]):
    if only:
        return [by_name(n) for n in only]
    return list(FIXTURE_SPECS)


def load_expected(fixture_name: str) -> dict:
    path = FIXTURES_DIR / fixture_name / "expected.json"
    return json.loads(path.read_text(encoding="utf-8"))


def print_summary(items: list[tuple[FixtureScores, Comparison]]) -> int:
    print()
    print("=" * 68)
    print(f"{'fixture':<28} {'status':<10} {'prev':>8} {'now':>8}  notes")
    print("-" * 68)
    regressions = 0
    failures = 0
    for scores, comp in items:
        glyph = {
            Status.NEW: "NEW",
            Status.UNCHANGED: "=",
            Status.IMPROVED: "▲",
            Status.REGRESSED: "▼",
            Status.FAILED: "FAILED",
        }[comp.status]
        prev = f"{comp.composite_prev:.4f}" if comp.composite_prev is not None else "—"
        notes = ""
        if comp.regression_reasons:
            notes = comp.regression_reasons[0]
        elif comp.improvement_notes:
            notes = comp.improvement_notes[0]
        print(f"{scores.fixture:<28} {glyph:<10} {prev:>8} "
              f"{scores.composite:>8.4f}  {notes}")
        if comp.status == Status.REGRESSED:
            regressions += 1
        if comp.status == Status.FAILED:
            failures += 1
    print("=" * 68)
    if failures:
        print(f"FAIL: {failures} fixture(s) failed structural checks.")
    if regressions:
        print(f"REGRESSED: {regressions} fixture(s) dropped beyond tolerance.")
    if not regressions and not failures:
        print("OK — no regressions.")
    return failures + regressions


def main() -> int:
    args = parse_args()
    specs = select_fixtures(args.only)

    run_results: list[tuple[FixtureScores, Comparison, Path, Path, Path | None]] = []
    items: list[tuple[FixtureScores, Comparison]] = []

    for spec in specs:
        fx_dir = FIXTURES_DIR / spec.name
        source_png = fx_dir / "source.png"
        expected = load_expected(spec.name)

        print(f"\n--- {spec.name} ---")
        t0 = time.time()
        try:
            pipeline = run_pipeline(spec.name, source_png, force=args.rerun_pipeline)
        except PipelineError as exc:
            print(f"  PIPELINE FAILED: {exc}")
            # Synthesize a FAILED scores record so the rest of the suite continues.
            failed_scores = FixtureScores(
                fixture=spec.name,
                structural={"ok": False, "slide_count": 0, "media_count": 0,
                            "text_run_count": 0, "placeholder_count": 0,
                            "zero_byte_media": 0,
                            "failures": [str(exc)]},
                text={"keyword_recall": 0.0, "cer": None, "extra_ratio": None,
                      "missing_keywords": expected.get("must_appear_text", []),
                      "found_keyword_count": 0, "total_text_chars": 0},
                visual={"exact_match_ratio": 0.0, "near_match_ratio": 0.0,
                        "blurred_match_ratio": 0.0, "color_histogram_sim": 0.0,
                        "available": False},
                counts={"textbox_count": 0, "textbox_in_range": False,
                        "textbox_expected": [0, 0], "image_object_count": 0,
                        "image_in_range": False, "image_expected": [0, 0]},
                composite=0.0,
                runtime_seconds=time.time() - t0,
                cached_pipeline=False,
            )
            comp = compare(failed_scores)
            items.append((failed_scores, comp))
            baseline_img = fx_dir / "baseline_reconstructed.png"
            run_results.append((
                failed_scores, comp, source_png, source_png,  # use source as placeholder
                baseline_img if baseline_img.exists() else None,
            ))
            continue
        if pipeline.cached:
            print(f"  pipeline: cached ({pipeline.work_dir.relative_to(REPO_ROOT)})")
        else:
            print(f"  pipeline: ran in {pipeline.runtime_seconds:.1f}s")

        source_text_ref = load_source_text_reference(pipeline.work_dir)
        scores = score_fixture(
            fixture_name=spec.name,
            source_png=source_png,
            expected=expected,
            pptx_path=pipeline.pptx_path,
            reconstructed_png=pipeline.preview_path,
            source_text_reference=source_text_ref,
            runtime_seconds=pipeline.runtime_seconds,
            cached_pipeline=pipeline.cached,
        )
        comp = compare(scores)
        print(f"  composite: {scores.composite:.4f}   status: {comp.status.value}")

        if args.update_baseline:
            write_baseline(scores, pipeline.preview_path)
            print(f"  baseline updated.")

        items.append((scores, comp))
        baseline_img = fx_dir / "baseline_reconstructed.png"
        run_results.append((
            scores, comp, source_png, pipeline.preview_path,
            baseline_img if baseline_img.exists() else None,
        ))

    exit_code = print_summary(items)

    if not args.no_report:
        report_path = render_report(run_results)
        print(f"\nReport: {report_path.relative_to(REPO_ROOT)}")
        print(f"  open {report_path}")

    return 1 if exit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
