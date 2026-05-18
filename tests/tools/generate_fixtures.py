"""Generate fixture source images via codex (HTML) + headless Chrome (PNG).

For each FixtureSpec in tests/tools/fixture_specs.py:
  1. Build a focused prompt from spec.design_brief.
  2. Invoke `codex exec` to write the slide HTML to disk.
  3. Render that HTML with headless Chrome at exactly spec.width × spec.height.
  4. Auto-generate expected.json from the spec.

Run:
    python tests/tools/generate_fixtures.py                 # all fixtures
    python tests/tools/generate_fixtures.py --only NAME     # one fixture
    python tests/tools/generate_fixtures.py --skip-codex    # only re-render existing HTML

Outputs (per fixture):
    tests/fixtures/<name>/source.html
    tests/fixtures/<name>/source.png
    tests/fixtures/<name>/expected.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.tools.fixture_specs import FIXTURE_SPECS, FixtureSpec, by_name

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chrome") or "",
]


def find_chrome() -> str:
    for cand in CHROME_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    raise SystemExit(
        "Could not find Chrome/Chromium. Install Google Chrome or set CHROME_PATH."
    )


def build_codex_prompt(spec: FixtureSpec, target_html: Path) -> str:
    return f"""\
You are generating a self-contained HTML file used as a test fixture for a slide
reconstruction pipeline. The HTML will be rendered by headless Chrome at exactly
{spec.width}x{spec.height} pixels and screenshotted to PNG. Treat it as a STATIC
poster — no JavaScript, no animations, no external network requests.

Hard requirements:
  - Write the complete HTML to: {target_html}
  - Use ONLY inline <style> and inline SVG. No <script>. No <link>. No remote fonts.
  - The <body> must be exactly {spec.width}px wide and {spec.height}px tall.
    Set body {{ margin: 0; width: {spec.width}px; height: {spec.height}px;
    overflow: hidden; }}.
  - All content must FIT inside the viewport — nothing clipped, no scrollbars.
  - Use system fonts only: -apple-system, "PingFang SC", "Hiragino Sans GB",
    "Microsoft YaHei", sans-serif.
  - No external images. Use inline SVG or pure CSS for any iconography / decoration.
  - Real text content must be present as actual text nodes (not as SVG <text> only),
    so a downstream OCR can read it.
  - Aim for a polished, visually rich slide that matches the brief precisely.
  - When the brief mentions specific text, reproduce that text verbatim.

Design brief:
\"\"\"
{spec.design_brief}
\"\"\"

When done, the file at {target_html} must be a complete, valid HTML5 document
that renders the described slide. Do not print or summarize the HTML — just
write the file.
"""


def run_codex(prompt: str, working_dir: Path) -> None:
    cmd = [
        "codex", "exec",
        "--skip-git-repo-check",
        "--sandbox", "workspace-write",
        "--cd", str(working_dir),
        "--color", "never",
        prompt,
    ]
    print(f"  $ codex exec ... (cwd={working_dir})", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"codex exec failed with code {result.returncode}")


def render_html_to_png(chrome: str, html_path: Path, png_path: Path,
                       width: int, height: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-sandbox",
        f"--window-size={width},{height}",
        f"--screenshot={png_path}",
        f"--force-device-scale-factor=1",
        f"file://{html_path}",
    ]
    print(f"  $ chrome --headless --screenshot ({width}x{height})", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not png_path.exists():
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"Chrome did not produce {png_path}")


def write_expected_json(spec: FixtureSpec, out_path: Path) -> None:
    payload = {
        "name": spec.name,
        "description": spec.description,
        "background": spec.background,
        "size": {"width": spec.width, "height": spec.height},
        "must_appear_text": spec.must_appear_text,
        "expected_image_objects": {
            "min": spec.expected_image_objects[0],
            "max": spec.expected_image_objects[1],
        },
        "expected_textboxes": {
            "min": spec.expected_textboxes[0],
            "max": spec.expected_textboxes[1],
        },
        "tags": spec.tags,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def html_looks_valid(html_path: Path, spec: FixtureSpec) -> bool:
    if not html_path.exists() or html_path.stat().st_size < 500:
        return False
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "<html" not in text.lower() or "</html>" not in text.lower():
        return False
    # Every must_appear_text fragment should be present in the HTML
    # (case-sensitive — these are the literal strings we expect downstream too).
    missing = [t for t in spec.must_appear_text if t not in text]
    if missing:
        print(f"    ! HTML is missing required text: {missing}", file=sys.stderr)
        return False
    return True


def generate_one(spec: FixtureSpec, *, skip_codex: bool, chrome: str) -> None:
    print(f"\n=== {spec.name} ({spec.width}x{spec.height}, {spec.background}) ===")
    fx_dir = FIXTURES_DIR / spec.name
    fx_dir.mkdir(parents=True, exist_ok=True)
    html_path = fx_dir / "source.html"
    png_path = fx_dir / "source.png"
    expected_path = fx_dir / "expected.json"

    if not skip_codex:
        prompt = build_codex_prompt(spec, html_path)
        run_codex(prompt, working_dir=fx_dir)

    if not html_looks_valid(html_path, spec):
        raise SystemExit(
            f"{spec.name}: source.html missing or incomplete after codex run.\n"
            f"  Inspect: {html_path}"
        )

    render_html_to_png(chrome, html_path, png_path, spec.width, spec.height)
    write_expected_json(spec, expected_path)
    print(f"  ✓ {png_path.relative_to(REPO_ROOT)} ({png_path.stat().st_size // 1024} KB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", action="append", default=[],
                   help="Generate only this fixture (repeatable).")
    p.add_argument("--skip-codex", action="store_true",
                   help="Skip HTML generation; re-render existing source.html.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chrome = find_chrome()
    specs = [by_name(n) for n in args.only] if args.only else FIXTURE_SPECS
    for spec in specs:
        generate_one(spec, skip_codex=args.skip_codex, chrome=chrome)
    print("\nAll fixtures generated.")


if __name__ == "__main__":
    main()
