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
        --pptx output/<run>/output/slides.pptx \\
        --out-dir output/<run>/output \\
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


def run(*, pptx: str, out_dir: str,
        dpi: int = 100, keep_pdf: bool = False,
        soffice: str | None = None,
        pdftoppm: str | None = None,
        verbose: bool = True) -> int:
    """Programmatic entry — same contract as the CLI flags.

    Calibration scripts call this directly to skip per-iteration Python
    startup (LibreOffice + pdftoppm themselves still fork, but that's
    intrinsic to the rendering toolchain).
    """
    pptx_path = Path(pptx).resolve()
    if not pptx_path.exists():
        sys.stderr.write(f"ERROR: pptx not found: {pptx_path}\n")
        return 1
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    previews_dir = out_dir_path / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    soffice_bin = which("soffice", soffice)
    pdftoppm_bin = which("pdftoppm", pdftoppm)

    if verbose:
        print(f"[render_preview] {pptx_path.name} → PDF via {soffice_bin}")
    subprocess.run(
        [soffice_bin, "--headless", "--convert-to", "pdf",
         "--outdir", str(out_dir_path), str(pptx_path)],
        check=True,
        env=fontconfig_env(),
        capture_output=not verbose,
    )
    pdf_path = out_dir_path / (pptx_path.stem + ".pdf")
    if not pdf_path.exists():
        sys.stderr.write(f"ERROR: PDF not produced at {pdf_path}\n")
        return 2

    if verbose:
        print(f"[render_preview] {pdf_path.name} → PNGs @ {dpi}dpi")
    subprocess.run(
        [pdftoppm_bin, "-png", "-r", str(dpi), str(pdf_path),
         str(previews_dir / "page")],
        check=True,
        capture_output=not verbose,
    )

    if not keep_pdf:
        pdf_path.unlink()
        if verbose:
            print(f"[render_preview] removed intermediate {pdf_path.name} "
                  "(use --keep-pdf to retain).")

    pngs = sorted(previews_dir.glob("page-*.png"))
    if verbose:
        print(f"[render_preview] {len(pngs)} previews in {previews_dir}")
    return 0


def main() -> int:
    args = parse_args()
    return run(pptx=args.pptx, out_dir=args.out_dir, dpi=args.dpi,
               keep_pdf=args.keep_pdf, soffice=args.soffice,
               pdftoppm=args.pdftoppm)


if __name__ == "__main__":
    raise SystemExit(main())
