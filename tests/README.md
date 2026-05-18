# DeckWeaver Regression Test Suite

Metric-based regression tests for the image→PPTX reconstruction pipeline.

Each run produces a **composite quality score per fixture** plus four
sub-category scores (structural / text / visual / counts). Scores are
compared against a per-fixture baseline; a fixture is "REGRESSED" when
any key metric drops more than its tolerance.

Pure scores, not pass/fail — the goal is to see whether a code change made
the reconstruction better or worse.

## Directory layout

```
tests/
├── fixtures/                       One folder per test case.
│   └── <fixture_name>/
│       ├── source.png              Ground-truth slide image (input to pipeline).
│       ├── source.html             HTML used to render source.png (reproducible).
│       ├── expected.json           Expectations (keywords, count ranges).
│       ├── baseline.json           Last recorded scores.
│       └── baseline_reconstructed.png
│                                   Last reconstructed preview (for 3-way diff).
├── tools/                          Fixture authoring helpers.
│   ├── fixture_specs.py            Spec for the 5 (or more) fixtures.
│   └── generate_fixtures.py        Spec → codex HTML → PNG → expected.json.
├── runner/                         Test logic.
│   ├── pipeline.py                 Runs convert.py per fixture, caches output.
│   ├── metrics.py                  Structural / text / visual / counts.
│   ├── baseline.py                 Per-fixture baseline load/save/compare.
│   ├── report.py                   HTML report: triptych, heatmaps, worst regions.
│   └── runner.py                   CLI entry.
├── reports/                        HTML reports per run (gitignored).
└── _work/                          Cached pipeline outputs (gitignored).
```

## Running

```bash
# All fixtures vs baseline (reuses cached pipeline output if source unchanged):
python tests/runner/runner.py

# Single fixture:
python tests/runner/runner.py -k light_cover_hero

# Force re-run of the pipeline (bypass cache, e.g. after changing scripts/):
python tests/runner/runner.py --rerun-pipeline

# Record current scores as new baseline:
python tests/runner/runner.py --update-baseline
```

After every run, see `tests/reports/<timestamp>/report.html` for the
visual diff — source / previous baseline / current side-by-side, plus
per-pixel heatmaps and top-3 worst regions with heuristic cause guesses.

## Metrics

| Category | Metrics |
|---|---|
| Structural | Slide count, no placeholder text, no zero-byte media — hard pass/fail |
| Text | Keyword recall, character error rate (CER), extra-text ratio |
| Visual | exact / near (Δ≤10) / blurred-near pixel match ratios, color histogram correlation |
| Counts | textbox count vs expected range, image-object count vs expected range |

Composite score weighs visual (0.40) + text (0.50) + counts (0.10). A
structural failure forces composite=0.

Regression tolerances are defined in `runner/baseline.py`. Default:
composite −0.01, blurred_match_ratio −0.02, color_histogram_sim −0.03,
keyword_recall −0.01.

## Adding a new fixture

1. Append a new `FixtureSpec(...)` to `tests/tools/fixture_specs.py`.
2. Run `python tests/tools/generate_fixtures.py --only <name>` — this calls
   codex to design HTML, renders it via headless Chrome to a PNG, and
   writes `expected.json` from the spec.
3. Inspect `tests/fixtures/<name>/source.png`. Tweak the spec if needed
   and regenerate.
4. Run `python tests/runner/runner.py -k <name> --update-baseline` to
   record initial scores.
5. Commit `tests/fixtures/<name>/`.

New fixtures without a baseline are reported as `NEW`, never as
regressions — adding cases never breaks the suite.

## Caches

- `tests/_work/<fixture>/` stores the latest pipeline output keyed by source
  PNG hash. Reused on subsequent runs if the source hasn't changed.
  Pass `--rerun-pipeline` to bypass.
- `tests/reports/<timestamp>/` keeps a timestamped HTML report per run.
  Both are gitignored.

## Dependencies

The suite reuses repo deps (`pillow`, `numpy`, `opencv-python`) — no new
installs needed. Fixture generation uses `codex` CLI + headless Chrome
(both already in dev setup).
