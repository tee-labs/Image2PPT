"""Run the DeckWeaver conversion pipeline on a single fixture.

By default the pipeline output is cached at tests/_work/<fixture>/
and reused on subsequent runs if the fixture's source.png has not
changed. Pass force=True (or --rerun-pipeline on the CLI) to
unconditionally re-run the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT_PY = REPO_ROOT / "scripts" / "convert.py"
WORK_ROOT = REPO_ROOT / "tests" / "_work"


class PipelineError(RuntimeError):
    """Raised when convert.py returns non-zero for a fixture."""


@dataclass
class PipelineResult:
    fixture_name: str
    work_dir: Path
    pptx_path: Path
    preview_path: Path        # First-page rendered preview
    runtime_seconds: float
    cached: bool              # True if reused from cache


def source_hash(source_png: Path) -> str:
    h = hashlib.sha256(source_png.read_bytes()).hexdigest()
    return h[:16]


def cache_marker_path(work_dir: Path) -> Path:
    return work_dir / ".source_hash"


def is_cache_valid(work_dir: Path, source_png: Path) -> bool:
    if not work_dir.exists():
        return False
    pptx = work_dir / "slides.pptx"
    if not pptx.exists():
        return False
    marker = cache_marker_path(work_dir)
    if not marker.exists():
        return False
    try:
        recorded = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return recorded == source_hash(source_png)


def first_preview_path(work_dir: Path) -> Path:
    previews = work_dir / "previews"
    if not previews.is_dir():
        raise FileNotFoundError(f"No previews directory under {work_dir}")
    pages = sorted(previews.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No page-*.png under {previews}")
    return pages[0]


def run_pipeline(fixture_name: str, source_png: Path, *,
                 force: bool = False, quiet: bool = False) -> PipelineResult:
    """Run convert.py for one fixture; return outputs.

    Cached at tests/_work/<fixture_name>/. Reused if source.png hash matches.
    """
    work_dir = WORK_ROOT / fixture_name

    if not force and is_cache_valid(work_dir, source_png):
        return PipelineResult(
            fixture_name=fixture_name,
            work_dir=work_dir,
            pptx_path=work_dir / "slides.pptx",
            preview_path=first_preview_path(work_dir),
            runtime_seconds=0.0,
            cached=True,
        )

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(CONVERT_PY),
        "--source", str(source_png),
        "--work-dir", str(work_dir),
        "--name", fixture_name,
    ]
    if not quiet:
        print(f"  $ convert.py --source {source_png.name} ...", flush=True)
    started = time.time()
    result = subprocess.run(
        cmd,
        capture_output=quiet,
        text=True,
        cwd=REPO_ROOT,
    )
    runtime = time.time() - started
    if result.returncode != 0:
        if quiet:
            sys.stderr.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        raise PipelineError(
            f"Pipeline failed for {fixture_name} (exit {result.returncode})"
        )

    cache_marker_path(work_dir).write_text(source_hash(source_png), encoding="utf-8")
    metadata = {
        "fixture": fixture_name,
        "source_png": str(source_png),
        "source_hash": source_hash(source_png),
        "runtime_seconds": round(runtime, 2),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (work_dir / ".pipeline_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return PipelineResult(
        fixture_name=fixture_name,
        work_dir=work_dir,
        pptx_path=work_dir / "slides.pptx",
        preview_path=first_preview_path(work_dir),
        runtime_seconds=runtime,
        cached=False,
    )
