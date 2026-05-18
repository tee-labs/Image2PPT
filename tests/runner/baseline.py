"""Per-fixture baseline storage and regression classification.

Each fixture stores its baseline alongside source.png:
  tests/fixtures/<name>/baseline.json                 - last recorded scores
  tests/fixtures/<name>/baseline_reconstructed.png    - last reconstructed preview

Calling write_baseline() updates both files.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tests.runner.metrics import FixtureScores

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Regression tolerances. A drop greater than the tolerance counts as a regression.
TOLERANCES = {
    "composite": 0.01,
    "visual.blurred_match_ratio": 0.02,
    "visual.color_histogram_sim": 0.03,
    "text.keyword_recall": 0.01,
}


class Status(str, Enum):
    NEW = "NEW"
    UNCHANGED = "UNCHANGED"
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    FAILED = "FAILED"  # structural failure


@dataclass
class Comparison:
    fixture: str
    status: Status
    composite_now: float
    composite_prev: float | None
    deltas: dict          # key -> (prev, now, delta)
    regression_reasons: list[str]
    improvement_notes: list[str]


def baseline_paths(fixture_name: str) -> tuple[Path, Path]:
    fx_dir = FIXTURES_DIR / fixture_name
    return fx_dir / "baseline.json", fx_dir / "baseline_reconstructed.png"


def load_baseline(fixture_name: str) -> dict | None:
    json_path, _ = baseline_paths(fixture_name)
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_baseline(scores: FixtureScores, reconstructed_png: Path) -> None:
    json_path, img_path = baseline_paths(scores.fixture)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(scores.to_dict(), ensure_ascii=False, indent=2,
                   sort_keys=True),
        encoding="utf-8",
    )
    if reconstructed_png.exists():
        shutil.copy2(reconstructed_png, img_path)


def _dig(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def compare(scores: FixtureScores) -> Comparison:
    """Compare current scores against the stored baseline."""
    if not scores.structural.get("ok"):
        return Comparison(
            fixture=scores.fixture,
            status=Status.FAILED,
            composite_now=scores.composite,
            composite_prev=None,
            deltas={},
            regression_reasons=scores.structural.get("failures", []),
            improvement_notes=[],
        )

    prev = load_baseline(scores.fixture)
    if prev is None:
        return Comparison(
            fixture=scores.fixture,
            status=Status.NEW,
            composite_now=scores.composite,
            composite_prev=None,
            deltas={},
            regression_reasons=[],
            improvement_notes=[],
        )

    deltas: dict[str, tuple] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    current = scores.to_dict()
    for key, tol in TOLERANCES.items():
        prev_v = _dig(prev, key)
        now_v = _dig(current, key)
        if prev_v is None or now_v is None:
            continue
        delta = now_v - prev_v
        deltas[key] = (prev_v, now_v, round(delta, 4))
        if delta < -tol:
            regressions.append(
                f"{key} dropped {prev_v:.4f} → {now_v:.4f} (Δ {delta:+.4f}, tol ±{tol})"
            )
        elif delta > tol:
            improvements.append(
                f"{key} improved {prev_v:.4f} → {now_v:.4f} (Δ {delta:+.4f})"
            )

    if regressions:
        status = Status.REGRESSED
    elif improvements:
        status = Status.IMPROVED
    else:
        status = Status.UNCHANGED

    return Comparison(
        fixture=scores.fixture,
        status=status,
        composite_now=scores.composite,
        composite_prev=prev.get("composite"),
        deltas=deltas,
        regression_reasons=regressions,
        improvement_notes=improvements,
    )
