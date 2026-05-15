#!/usr/bin/env python
"""Mark OCR entries that look like template logos as preserve_visual=true.

Two complementary mechanisms:

  B. Cross-page repeat detection. If a text string occurs on at least
     `--min-ratio` of pages, all at roughly the same bbox position
     (range ≤ `--pos-tolerance` in both axes), treat every occurrence as
     a template logo. Catches header/footer logos, university names,
     recurring brand wordmarks — anything that lives at the same place
     on most slides.

  C. User-provided whitelist. A plain text file (one entry per line,
     `#` for comments) of exact text strings to always preserve. Useful
     for one-off logos that only appear on a few pages and so escape
     mechanism B, or to seed the preservation list from prior knowledge
     of the deck.

Sets `preserve_visual: true` on matched OCR items in place. Downstream
(erase_text, build_inventory, inventory_to_layout) already honours this
flag via `should_preserve_visual` — text stays as pixels, never
becomes an editable text element, never gets erased.

Run AFTER `ocr_review_apply.py` and BEFORE `run_pipeline.py`:

    python scripts/optional/find_template_logos.py --work-dir <run_dir> \\
        [--whitelist logo_whitelist.txt] \\
        [--min-ratio 0.4] [--pos-tolerance 30]

Idempotent — running twice yields the same flags.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-dir", required=True,
                   help="Run directory containing inventory/page_*.ocr.json.")
    p.add_argument("--whitelist", default=None,
                   help="Optional whitelist file: one text string per line, "
                        "lines starting with # are comments. Matches are "
                        "exact (after .strip()).")
    p.add_argument("--min-ratio", type=float, default=0.4,
                   help="Cross-page repeat threshold. A text must appear on "
                        "≥ this fraction of pages to be flagged. "
                        "Default 0.4.")
    p.add_argument("--pos-tolerance", type=int, default=30,
                   help="Maximum bbox-center range (max-min across pages) "
                        "in pixels, in BOTH axes, for a recurring text to "
                        "be considered a stable template logo. Default 30.")
    p.add_argument("--min-pages", type=int, default=2,
                   help="Absolute minimum page count regardless of ratio. "
                        "Default 2 — a logo on a single page can't be "
                        "detected by mechanism B; use --whitelist for those.")
    p.add_argument("--max-text-length", type=int, default=20,
                   help="Maximum weighted text length (CJK char = 2 units, "
                        "ASCII char = 1 unit) for an entry to be considered "
                        "a logo by mechanism B. Default 20 = up to 10 "
                        "Chinese characters or 20 English characters. Long "
                        "strings that happen to repeat (e.g. footer "
                        "citations, data sources) are not logos. The user "
                        "whitelist (mechanism C) is NOT subject to this "
                        "filter.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be flagged without writing back.")
    p.add_argument("--print-candidates", action="store_true",
                   help="Print all texts that repeat on ≥2 pages (regardless "
                        "of position stability), to help build a whitelist.")
    return p.parse_args()


def load_pages(work: Path) -> dict[str, list[dict]]:
    pages: dict[str, list[dict]] = {}
    for p in sorted((work / "inventory").glob("page_*.ocr.json")):
        # `page_NN.ocr.json` → key `NN`. Skip the `ocr.raw.json` backups.
        if p.name.endswith(".raw.json"):
            continue
        page_id = p.stem.split("_", 1)[1].split(".", 1)[0]
        pages[page_id] = json.loads(p.read_text(encoding="utf-8"))
    return pages


def weighted_length(text: str) -> int:
    """Weighted character length: CJK char = 2, ASCII/other = 1.

    A 10-character Chinese logo maps to 20, comparable to a medium-length
    Latin-script logo. This lets one threshold serve both scripts.
    """
    total = 0
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs + Extension A + Compatibility Ideographs
        # + Hiragana + Katakana + Hangul (treat all wide scripts as 2)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF
            or 0x3040 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7A3
            or 0xFF00 <= cp <= 0xFFEF  # fullwidth punctuation
        ):
            total += 2
        else:
            total += 1
    return total


def load_whitelist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    raw = path.read_text(encoding="utf-8").splitlines()
    return {
        line.strip() for line in raw
        if line.strip() and not line.lstrip().startswith("#")
    }


def find_template_logos(
    pages: dict[str, list[dict]],
    min_ratio: float,
    min_pages: int,
    pos_tolerance: int,
    max_text_length: int,
) -> tuple[dict[str, set[int]], list[tuple]]:
    """Detect template logos by cross-page text + position stability.

    Returns (matched, summary) where matched[page_id] is the set of OCR
    indices to flag, and summary is a list of human-readable rows for
    stdout.
    """
    total_pages = len(pages)
    threshold = max(min_pages, int(round(min_ratio * total_pages)))

    text_occ: dict[str, list[tuple]] = defaultdict(list)
    for page_id, entries in pages.items():
        for idx, item in enumerate(entries):
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            if weighted_length(text) > max_text_length:
                continue
            cx = (int(item["x1"]) + int(item["x2"])) / 2.0
            cy = (int(item["y1"]) + int(item["y2"])) / 2.0
            text_occ[text].append((page_id, idx, cx, cy))

    matched: dict[str, set[int]] = defaultdict(set)
    summary: list[tuple] = []
    for text, occs in text_occ.items():
        unique_pages = {o[0] for o in occs}
        if len(unique_pages) < threshold:
            continue
        xs = [o[2] for o in occs]
        ys = [o[3] for o in occs]
        if max(xs) - min(xs) > pos_tolerance:
            continue
        if max(ys) - min(ys) > pos_tolerance:
            continue
        for page_id, idx, _, _ in occs:
            matched[page_id].add(idx)
        summary.append((
            text, len(unique_pages), total_pages,
            int(statistics.median(xs)), int(statistics.median(ys)),
        ))
    summary.sort(key=lambda r: (-r[1], r[3], r[4]))
    return matched, summary


def apply_whitelist(
    pages: dict[str, list[dict]],
    whitelist: set[str],
) -> tuple[dict[str, set[int]], list[tuple]]:
    """Flag OCR entries whose text exactly matches a whitelist string."""
    matched: dict[str, set[int]] = defaultdict(set)
    summary: list[tuple] = []
    if not whitelist:
        return matched, summary
    for page_id, entries in pages.items():
        for idx, item in enumerate(entries):
            text = str(item.get("text", "") or "").strip()
            if text in whitelist:
                matched[page_id].add(idx)
                summary.append((page_id, idx, text))
    summary.sort()
    return matched, summary


def print_candidates(pages: dict[str, list[dict]]) -> None:
    """Print texts that repeat on ≥2 pages, sorted by frequency, to help
    the user populate a whitelist. Includes position range so the user
    can tell stable logos from incidental text repeats.
    """
    text_occ: dict[str, list[tuple]] = defaultdict(list)
    for page_id, entries in pages.items():
        for item in entries:
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            cx = (int(item["x1"]) + int(item["x2"])) / 2.0
            cy = (int(item["y1"]) + int(item["y2"])) / 2.0
            text_occ[text].append((page_id, cx, cy))

    rows = []
    for text, occs in text_occ.items():
        pages_set = {o[0] for o in occs}
        if len(pages_set) < 2:
            continue
        xs = [o[1] for o in occs]
        ys = [o[2] for o in occs]
        rows.append((
            text, len(pages_set), len(pages),
            int(max(xs) - min(xs)), int(max(ys) - min(ys)),
        ))
    rows.sort(key=lambda r: (-r[1], r[3] + r[4]))
    print(f"{'text':<40} {'pages':>6} {'xrange':>8} {'yrange':>8}")
    for text, npg, total, xr, yr in rows:
        disp = text if len(text) <= 38 else text[:35] + "..."
        print(f"{disp:<40} {npg:>3}/{total:<3} {xr:>8} {yr:>8}")


def main() -> int:
    args = parse_args()
    work = Path(args.work_dir)
    pages = load_pages(work)
    if not pages:
        print(f"no inventory/page_*.ocr.json found under {work}")
        return 1

    if args.print_candidates:
        print_candidates(pages)
        return 0

    whitelist = load_whitelist(Path(args.whitelist)) if args.whitelist else set()

    repeat_matched, repeat_sum = find_template_logos(
        pages, args.min_ratio, args.min_pages, args.pos_tolerance,
        args.max_text_length,
    )
    wl_matched, wl_sum = apply_whitelist(pages, whitelist)

    # Merge: union of indices per page.
    merged: dict[str, set[int]] = defaultdict(set)
    for d in (repeat_matched, wl_matched):
        for page_id, idxs in d.items():
            merged[page_id].update(idxs)

    print(f"== Mechanism B: cross-page repeat detection "
          f"(min_ratio={args.min_ratio}, min_pages={args.min_pages}, "
          f"pos_tolerance={args.pos_tolerance}px, "
          f"max_text_length={args.max_text_length}, "
          f"total_pages={len(pages)}) ==")
    if repeat_sum:
        print(f"{'text':<40} {'pages':>8} {'med_x':>6} {'med_y':>6}")
        for text, n, total, mx, my in repeat_sum:
            disp = text if len(text) <= 38 else text[:35] + "..."
            print(f"{disp:<40} {n:>3}/{total:<3}   {mx:>6} {my:>6}")
    else:
        print("  (no template logos detected)")

    print()
    print(f"== Mechanism C: user whitelist "
          f"({args.whitelist or '<none>'}, {len(whitelist)} entries) ==")
    if wl_sum:
        # Group counts by text.
        per_text: dict[str, int] = defaultdict(int)
        for _, _, text in wl_sum:
            per_text[text] += 1
        for text, n in sorted(per_text.items(), key=lambda kv: (-kv[1], kv[0])):
            disp = text if len(text) <= 38 else text[:35] + "..."
            print(f"  {disp:<40} matched {n} entries")
    else:
        print("  (no whitelist matches)")

    total_flags = sum(len(s) for s in merged.values())
    print()
    print(f"== Will flag {total_flags} OCR entries across "
          f"{len(merged)} pages as preserve_visual=true ==")

    if args.dry_run:
        return 0

    # Write back.
    for page_id, entries in pages.items():
        idxs = merged.get(page_id, set())
        if not idxs:
            continue
        for i in idxs:
            entries[i]["preserve_visual"] = True
        out_path = work / "inventory" / f"page_{page_id}.ocr.json"
        out_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"wrote updates to {len(merged)} ocr.json files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
