#!/usr/bin/env python
"""Render a built PPTX into a PDF + per-page PNG previews for QA.

Wraps LibreOffice (`soffice --headless`) for PPTX → PDF, then `pdftoppm`
for PDF → per-page PNGs. The skill's QA step is "compare reconstructed
slide previews against source images" — this script makes that one
command instead of two ad-hoc shell invocations the agent has to recall.

Both tools are required:
- LibreOffice — `brew install --cask libreoffice` (macOS) or
  `apt install libreoffice` (Linux). Headless mode only.
- Poppler/pdftoppm — `brew install poppler` or `apt install poppler-utils`.

Usage:
    python scripts/verify/render_preview.py \\
        --pptx output_project/<run>/output/slides.pptx \\
        --out-dir output_project/<run>/output \\
        [--dpi 100] [--keep-pdf]

Writes:
    <out_dir>/slides.pdf             (kept if --keep-pdf, removed otherwise)
    <out_dir>/previews/page-NN.png   (one per slide, zero-padded)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from fontconfig_helper import fontconfig_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPTX → PDF → per-page PNG previews.")
    p.add_argument("--pptx", required=True, help="Built .pptx to render.")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write slides.pdf + previews/ into.")
    p.add_argument("--dpi", type=int, default=100,
                   help="Preview PNG resolution (default 100).")
    p.add_argument("--keep-pdf", action="store_true",
                   help="Keep the intermediate PDF (default: deleted after "
                        "PNG conversion).")
    p.add_argument("--soffice", default=None,
                   help="Path to soffice binary (default: auto-detect).")
    p.add_argument("--pdftoppm", default=None,
                   help="Path to pdftoppm binary (default: auto-detect).")
    return p.parse_args()


def which(name: str, override: str | None) -> str:
    if override:
        return override
    found = shutil.which(name)
    if not found:
        # macOS LibreOffice ships inside an .app bundle that may not be on PATH.
        if name == "soffice":
            mac_default = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            if Path(mac_default).exists():
                return mac_default
        raise SystemExit(f"ERROR: {name!r} not found on PATH. "
                         f"Install it or pass --{name}.")
    return found


def main() -> int:
    args = parse_args()
    pptx = Path(args.pptx).resolve()
    if not pptx.exists():
        sys.stderr.write(f"ERROR: pptx not found: {pptx}\n")
        return 1
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    soffice = which("soffice", args.soffice)
    pdftoppm = which("pdftoppm", args.pdftoppm)

    print(f"[render_preview] {pptx.name} → PDF via {soffice}")
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf",
         "--outdir", str(out_dir), str(pptx)],
        check=True,
        env=fontconfig_env(),
    )
    pdf_path = out_dir / (pptx.stem + ".pdf")
    if not pdf_path.exists():
        sys.stderr.write(f"ERROR: PDF not produced at {pdf_path}\n")
        return 2

    print(f"[render_preview] {pdf_path.name} → PNGs @ {args.dpi}dpi")
    subprocess.run(
        [pdftoppm, "-png", "-r", str(args.dpi), str(pdf_path),
         str(previews_dir / "page")],
        check=True,
    )

    if not args.keep_pdf:
        pdf_path.unlink()
        print(f"[render_preview] removed intermediate {pdf_path.name} "
              "(use --keep-pdf to retain).")

    pngs = sorted(previews_dir.glob("page-*.png"))
    print(f"[render_preview] {len(pngs)} previews in {previews_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
