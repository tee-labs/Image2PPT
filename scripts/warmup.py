#!/usr/bin/env python
"""Pre-download every model the skill needs, so the first real run isn't
mixed up with cold-cache downloads.

Touches two caches:

1. **PaddleOCR PP-OCRv5** (~85 MB det + rec, cached under
   `~/.paddlex/official_models/`). Triggered by instantiating
   `PaddleOCR()` and running it once on a synthesized small image.
2. **RMBG-1.4 ONNX FP16** (~88 MB, cached under
   `~/.cache/huggingface/hub/`). Optional — skip with `--skip-rmbg` if
   you don't plan to run `rmbg_postprocess.py`.

Usage:
    python scripts/warmup.py [--skip-rmbg] [--skip-paddle]

Exits non-zero on any failure so a `bootstrap.sh` script can fail fast.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.gpu import describe as describe_devices, paddle_device  # noqa: E402


def banner(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def warmup_paddle() -> None:
    banner("PaddleOCR PP-OCRv5 (det + rec)")
    warnings.filterwarnings("ignore")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("FLAGS_print_log", "0")
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise SystemExit(
            "paddleocr is not installed. Run "
            "`pip install 'paddleocr>=3' 'paddlex[ocr]'` first."
        )
    import numpy as np
    from PIL import Image
    t0 = time.time()
    ocr = PaddleOCR(
        lang="ch",
        device=paddle_device(),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Synthesize a tiny image — content doesn't matter, we just want
    # PaddleOCR to do its first-run model load.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dummy.png"
        Image.fromarray(
            (np.random.rand(64, 256, 3) * 255).astype("uint8")
        ).save(p)
        ocr.predict(str(p))
    print(f"  ok ({time.time() - t0:.1f}s)")


def warmup_rmbg() -> None:
    banner("RMBG-1.4 ONNX")
    try:
        from huggingface_hub import hf_hub_download
        import onnxruntime  # noqa: F401 — just verify it imports.
    except ImportError:
        raise SystemExit(
            "huggingface_hub / onnxruntime not installed. Run "
            "`pip install onnxruntime huggingface_hub` first."
        )
    t0 = time.time()
    repo = os.environ.get("RMBG_REPO", "briaai/RMBG-1.4")
    path = hf_hub_download(repo, "onnx/model_fp16.onnx")
    print(f"  ok ({time.time() - t0:.1f}s)\n  cached at: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-download every model the skill uses.")
    p.add_argument("--skip-paddle", action="store_true",
                   help="Skip PaddleOCR warmup.")
    p.add_argument("--skip-rmbg", action="store_true",
                   help="Skip RMBG warmup (optional model — only needed "
                        "for rmbg_postprocess.py).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(describe_devices(), flush=True)
    steps = [
        (args.skip_paddle, warmup_paddle),
        (args.skip_rmbg, warmup_rmbg),
    ]
    for skip, fn in steps:
        if skip:
            continue
        try:
            fn()
        except SystemExit:
            raise
        except Exception as exc:
            print(f"  FAIL: {exc}", file=sys.stderr)
            return 1
    banner("done")
    print("All requested models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
