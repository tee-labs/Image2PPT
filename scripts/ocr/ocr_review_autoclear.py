#!/usr/bin/env python
"""Auto-clear obvious spurious OCR detections before agent review.

Runs against an ocr_review.json (produced by prepare_ocr.py) and
sets corrected_text="" on entries that match conservative heuristics
for "this is not real editable text" — logo carvings, page-edge icon
glyphs, decorative punctuation, etc. The agent only needs to review
whatever's left.

This module is also imported by prepare_ocr.py so the same rules run
inline as Stage 3 of the OCR prep — running this standalone is only
needed for ad-hoc re-clears after manual edits.

The rules below are intentionally conservative: they target patterns
that are almost always decorative or logo-like rather than real editable
text.

Usage:
    python scripts/ocr/ocr_review_autoclear.py \\
        --review path/to/page_NN.ocr_review.json \\
        --image-size 1280x720 \\
        [--dry-run]

The script writes corrected_text="" in-place on matched entries and
adds a `notes` field naming the rule that fired. The agent can still
override any decision by editing corrected_text manually.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CJK_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in CJK_RANGES)


def has_cjk(s: str) -> bool:
    return any(is_cjk(c) for c in s)


# Each rule: (name, predicate(text, conf, w, h, rx, ry) -> bool).

def H1_logo_strip(t, conf, w, h, rx, ry):
    """Likely brand/logo tokens in top/bottom strip OR page side-margins."""
    keys = (
        "UNIVERSITY", "UNNERTY", "INSTITUTE", "COLLEGE", "GROUP",
        "CORP", "LTD", "INC", "LAB", "LOGO",
    )
    if not any(k in t for k in keys):
        return False
    if ry < 0.15 or ry > 0.85:
        return True
    # Side-margin extension catches logos that sit mid-page near a
    # vertical edge.
    return rx < 0.16 or rx > 0.85


def H2_single_cjk_lowconf(t, conf, w, h, rx, ry):
    """A single CJK char with confidence under 0.55 — almost always an
    icon glyph misread (e.g. 皿, 山, 自 for a building/chart pictogram)."""
    s = t.strip()
    return len(s) == 1 and is_cjk(s) and conf < 0.55


def H3_punct_symbol_only(t, conf, w, h, rx, ry):
    """Short string consisting entirely of punctuation/symbols (no
    alphanumeric, no CJK) with confidence < 0.80 — decorative quotes,
    stray ellipses, dingbats."""
    s = t.strip()
    if not s or "…" in s or s in {".", ".."}:
        return False
    if any(c.isalnum() or is_cjk(c) for c in s):
        return False
    return len(s) <= 4 and conf < 0.80


def H4_arrow_dash_fragment(t, conf, w, h, rx, ry):
    """Arrow/dash fragments (←, →, ---, -->) caught at confidence < 0.90."""
    s = t.strip()
    return bool(re.fullmatch(r"[-=>→←↑↓·•]+", s)) and len(s) <= 6 and conf < 0.90


def H5_short_thin_lowconf(t, conf, w, h, rx, ry):
    """Non-CJK short text in a thin bbox (h ≤ 15) with conf < 0.95 — text
    so small it's almost certainly part of a logo or icon, not a real label.
    Carve-out: bottom-right corner page numbers (rx>0.95, ry>0.92) are
    legitimate."""
    s = t.strip()
    if not s or has_cjk(s) or "…" in s or s == ".":
        return False
    if len(s) > 8 or h > 15 or conf >= 0.95:
        return False
    if rx > 0.95 and ry > 0.92:
        return False
    return True


def H6_logo_digits_top_corner(t, conf, w, h, rx, ry):
    """A 1-3 char token of digits + a few digit-like letters (O, D, I, l,
    Z), in the top corners of the page — typically OCR catching a fragment
    of a stamp/seal ring like "1909" or "00"."""
    s = t.strip()
    if not s or has_cjk(s):
        return False
    if not re.fullmatch(r"[0-9OoDIlZ]{1,3}", s):
        return False
    return ry < 0.12 and (rx > 0.80 or rx < 0.15) and conf < 0.95


RULES = [
    ("H1_logo_strip", H1_logo_strip),
    ("H2_single_cjk_lowconf", H2_single_cjk_lowconf),
    ("H3_punct_symbol_only", H3_punct_symbol_only),
    ("H4_arrow_dash_fragment", H4_arrow_dash_fragment),
    ("H5_short_thin_lowconf", H5_short_thin_lowconf),
    ("H6_logo_digits_top_corner", H6_logo_digits_top_corner),
]


def match_rule(text, conf, bbox, image_w, image_h):
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = cx / max(1, image_w), cy / max(1, image_h)
    for name, pred in RULES:
        try:
            if pred(text, conf, w, h, rx, ry):
                return name
        except Exception:
            continue
    return None


def parse_size(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", s.strip())
    if not m:
        raise SystemExit(f"--image-size expects WxH (e.g. 1280x720), got {s!r}")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--review", required=True,
                   help="Review JSON from prepare_ocr.py (modified in place).")
    p.add_argument("--image-size", default="1280x720",
                   help="Source slide size WxH (default 1280x720).")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write the file — just print what would change.")
    args = p.parse_args()

    W, H = parse_size(args.image_size)
    path = Path(args.review)
    data = json.loads(path.read_text(encoding="utf-8"))

    cleared = 0
    by_rule: dict[str, int] = {}
    for e in data.get("entries", []):
        if e.get("corrected_text") is not None:
            continue
        rule = match_rule(
            e["original_text"], float(e["confidence"]),
            tuple(e["bbox"]), W, H,
        )
        if rule:
            e["corrected_text"] = ""
            e["notes"] = f"auto-cleared by {rule}"
            cleared += 1
            by_rule[rule] = by_rule.get(rule, 0) + 1

    total = len(data.get("entries", []))
    remaining = total - cleared
    print(f"[ocr_review_autoclear] {cleared}/{total} entries auto-cleared, "
          f"{remaining} left for agent review.")
    for rule, n in sorted(by_rule.items()):
        print(f"  {rule}: {n}")

    if not args.dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
