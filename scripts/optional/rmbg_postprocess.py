#!/usr/bin/env python
"""Post-process inventory asset PNGs with RMBG to add transparent bgs.

The cv2 detector emits rectangular RGB asset PNGs whose host-slide
background colour is baked into the rectangle. Pasting such an asset
onto a different template makes the rectangle visible as a colour-
block halo. RMBG-1.4 (briaai/RMBG-1.4, ONNX FP16, ~88 MB) is a U-net
background remover that takes an image and returns an alpha mask;
this script applies it to every "icon-like" asset in a layout's
asset directory and overwrites the PNG with an RGBA version.
RMBG-2.0 is gated on Hugging Face — set RMBG_REPO=briaai/RMBG-2.0 to
use it once you've accepted the EULA.

When to skip an asset:
- bbox too big (>0.5 of slide area): probably a background card, not
  an icon; transparency would chew through the legible body.
- already RGBA: a sub-icon / outline-masked asset already has alpha.
- bbox doesn't look icon-shaped (aspect ratio outside 0.3..3): a wide
  banner or tall list isn't what RMBG was trained for.

Usage:
    python scripts/optional/rmbg_postprocess.py \\
        --assets-dir output_project/<run_dir>/assets/page_NN \\
        --layout    output_project/<run_dir>/layouts/page_NN.layout.json

The layout JSON is read for per-element bbox so we can filter sensibly.
Each qualifying PNG is overwritten in place with an RGBA version.

License note: RMBG-2.0 is licensed for non-commercial use. The model
weights are downloaded from Hugging Face on first run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_SESSION = None
_INPUT_NAME = None


def _load_session():
    """Lazy-load the RMBG ONNX session."""
    global _SESSION, _INPUT_NAME
    if _SESSION is not None:
        return _SESSION
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit("RMBG postprocess needs onnxruntime: "
                         "pip install --user onnxruntime") from e
    import os
    from huggingface_hub import hf_hub_download
    repo = os.environ.get("RMBG_REPO", "briaai/RMBG-1.4")
    model_path = hf_hub_download(repo, "onnx/model_fp16.onnx")
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    _SESSION = ort.InferenceSession(model_path, providers=providers)
    _INPUT_NAME = _SESSION.get_inputs()[0].name
    return _SESSION


def _preprocess(img_bgr: np.ndarray, size: int = 1024) -> np.ndarray:
    """RMBG-1.4: 1024x1024 RGB float32, [-1, 1] normalised. The .fp16
    ONNX file just has fp16 weights; its inputs are still float32."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = img.astype(np.float32) / 127.5 - 1.0
    arr = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return arr


def remove_background(img_bgr: np.ndarray) -> np.ndarray:
    """Return an alpha mask (uint8, same H×W as input) where higher
    values mean more opaque (foreground). RMBG-1.4 outputs a sigmoid
    map directly, no extra activation needed."""
    sess = _load_session()
    h, w = img_bgr.shape[:2]
    inp = _preprocess(img_bgr)
    out = sess.run(None, {_INPUT_NAME: inp})[0]
    if isinstance(out, list):
        out = out[-1]
    mask = out[0, 0]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return (np.clip(mask, 0, 1) * 255).astype(np.uint8)


def is_icon_like(box: list[int], slide_area: int) -> bool:
    """Heuristic: small enough to be an icon AND not a full-width banner."""
    x, y, w, h = box
    if w < 24 or h < 24:
        return False
    if w * h > 0.4 * slide_area:
        return False
    aspect = w / max(1, h)
    if aspect < 0.25 or aspect > 4.0:
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RMBG postprocess for assets")
    p.add_argument("--assets-dir", required=True,
                   help="Directory of asset PNGs.")
    p.add_argument("--layout", required=True,
                   help="layout.json for slide_size/source_width/height "
                        "and per-element bbox.")
    p.add_argument("--skip-rgba", action="store_true", default=True,
                   help="Skip PNGs that already have alpha "
                        "(default: True).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    layout = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    sw = layout.get("source_width", 1280)
    sh = layout.get("source_height", 720)
    slide_area = sw * sh
    assets_dir = Path(args.assets_dir)

    processed = 0
    skipped = 0
    for el in layout.get("elements", []):
        if el.get("type") != "image":
            continue
        path = el.get("path", "")
        # path looks like "assets/page_NN/v005.png"; the file lives in
        # assets_dir under just the basename.
        png = assets_dir / Path(path).name
        if not png.exists():
            continue
        if not is_icon_like(el["box"], slide_area):
            skipped += 1
            continue
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if args.skip_rgba and img.ndim == 3 and img.shape[2] == 4:
            skipped += 1
            continue
        if img.shape[2] == 4:
            img = img[:, :, :3]
        try:
            alpha = remove_background(img)
        except Exception as e:
            print(f"  {png.name}: RMBG failed ({e})", file=sys.stderr)
            continue
        rgba = np.dstack([img, alpha])
        cv2.imwrite(str(png), rgba)
        processed += 1
    print(json.dumps({"processed": processed, "skipped": skipped},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
