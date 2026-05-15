#!/usr/bin/env python
"""Heuristic table-region detector driven by OCR layout.

`table_recognize.py` is structure-only — it ASSUMES its input is a table.
Asking the agent to identify tables visually works but doesn't scale, and
the auto-extract pipeline currently has no automatic entry point into the
table flow. This script bridges that gap: it scans the OCR JSON for a
rectangular grid of texts (≥`min_rows` × ≥`min_cols`, mutually aligned
on both axes) and emits candidate bboxes that `table_recognize.py` can
then verify with SLANet_plus.

Approach (no model — pure layout heuristic):

  1. Cluster OCR detections into rows by y-center (tolerance scales with
     the median glyph height so a 12pt note and a 24pt cell still cluster
     correctly within their own row).
  2. Inside each row, accept rows that contain ≥`min_cols` separable text
     items (gap between consecutive items > 0.5 × median glyph height).
  3. Cluster ROW-LOCAL column positions (left edge of each item) across
     rows. Accept a column position if it's hit by ≥`min_rows` rows.
  4. Group consecutive accepted rows whose accepted columns share the
     same set of column-positions (±tolerance) — that's one table.

The detector intentionally favours *precision over recall*: false
positives are expensive because they downgrade legitimate stat callouts
or icon labels into a fake table render. Real tables on a slide are
usually clean, well-aligned and have ≥3 rows and ≥2 columns — exactly
the regime the heuristic targets.

The bbox emitted padds the tight cell-union by 6 px so SLANet_plus has
a small margin around the cells (its model was trained on tables with
some surrounding whitespace).

Usage:

    python scripts/tables/detect_tables.py \
        --ocr inventory/page_NN.ocr.json \
        --image-size 1280x720 \
        --out inventory/page_NN.table_candidates.json

Output schema:
    {
      "candidates": [
        {
          "bbox": [x1, y1, x2, y2],
          "rows": 4, "cols": 3,
          "cells_used": 12
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heuristic table-region detector.")
    p.add_argument("--ocr", required=True, help="OCR JSON path.")
    p.add_argument("--image-size",
                   help="WxH of the source image. Used only for sanity "
                        "checks; the script doesn't read pixels.")
    p.add_argument("--out", help="Output JSON path (default: stdout).")
    p.add_argument("--min-rows", type=int, default=3)
    p.add_argument("--min-cols", type=int, default=2)
    p.add_argument("--col-tolerance-px", type=int, default=20,
                   help="Two row entries are in the SAME column when their "
                        "x-lefts differ by ≤ this many pixels.")
    p.add_argument("--padding-px", type=int, default=6,
                   help="Padding around the bbox to give SLANet a small "
                        "margin around the cells.")
    return p.parse_args()


def _row_cluster(items: list[dict]) -> list[list[dict]]:
    """Cluster OCR items into rows by y-center, tolerance ≈ glyph height."""
    if not items:
        return []
    enriched = []
    for it in items:
        y1, y2 = int(it["y1"]), int(it["y2"])
        enriched.append({**it, "_cy": (y1 + y2) / 2.0, "_h": y2 - y1})
    enriched.sort(key=lambda r: r["_cy"])
    heights = [r["_h"] for r in enriched if r["_h"] > 0]
    if not heights:
        return []
    tol = max(8.0, 0.6 * median(heights))
    rows: list[list[dict]] = []
    current: list[dict] = [enriched[0]]
    current_y = enriched[0]["_cy"]
    for r in enriched[1:]:
        if abs(r["_cy"] - current_y) <= tol:
            current.append(r)
            # Rolling y so a row that drifts slightly downward still merges.
            current_y = sum(c["_cy"] for c in current) / len(current)
        else:
            rows.append(current)
            current = [r]
            current_y = r["_cy"]
    rows.append(current)
    return rows


def _row_columns(row: list[dict]) -> list[int]:
    """Sorted left-edges of items in this row, after dropping items that
    visually overlap (overlapping bboxes are one cell with internal line
    breaks, not two cells)."""
    row = sorted(row, key=lambda r: int(r["x1"]))
    cols: list[int] = []
    last_x2 = -10_000
    for r in row:
        x1, x2 = int(r["x1"]), int(r["x2"])
        if x1 < last_x2 + 4:
            # Overlaps the previous cell — same column, just a wrapped line.
            last_x2 = max(last_x2, x2)
            continue
        cols.append(x1)
        last_x2 = x2
    return cols


def _column_clusters(rows: list[list[dict]],
                     tolerance: int) -> list[int]:
    """Find x-positions that appear in many rows; each cluster center is
    one column. A cluster is accepted later only if enough rows hit it."""
    xs: list[int] = []
    for row in rows:
        xs.extend(_row_columns(row))
    if not xs:
        return []
    xs.sort()
    clusters: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [int(round(sum(c) / len(c))) for c in clusters]


def _row_hits(row: list[dict],
              column_centers: list[int],
              tolerance: int) -> set[int]:
    """Indices of column centers that THIS row populates."""
    row_xs = _row_columns(row)
    hits: set[int] = set()
    for x in row_xs:
        # Snap each row item to the nearest column center within tolerance.
        best: int | None = None
        best_d = tolerance + 1
        for j, cx in enumerate(column_centers):
            d = abs(x - cx)
            if d < best_d:
                best = j
                best_d = d
        if best is not None:
            hits.add(best)
    return hits


def _coeff_var(values: list[float]) -> float:
    """Coefficient of variation. Returns +inf for an empty list."""
    if not values:
        return float("inf")
    mean = sum(values) / len(values)
    if mean <= 0:
        return float("inf")
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / mean


def detect(items: list[dict],
           min_rows: int,
           min_cols: int,
           col_tolerance: int,
           padding: int,
           max_col_width_cv: float = 0.6,
           max_row_height_cv: float = 0.6,
           min_density: float = 0.6) -> list[dict]:
    if not items:
        return []
    rows = _row_cluster(items)
    if len(rows) < min_rows:
        return []
    centers = _column_clusters(rows, col_tolerance)
    if len(centers) < min_cols:
        return []

    # For each column center, how many rows hit it? Drop rare columns
    # (hit by <min_rows rows) — those are isolated outliers, not table
    # columns. Then drop rows that hit <min_cols of the surviving columns.
    hits_by_row = [
        _row_hits(row, centers, col_tolerance) for row in rows
    ]
    col_hit_count = [0] * len(centers)
    for h in hits_by_row:
        for j in h:
            col_hit_count[j] += 1
    surviving_cols = {j for j, c in enumerate(col_hit_count) if c >= min_rows}
    if len(surviving_cols) < min_cols:
        return []

    row_valid: list[bool] = []
    for h in hits_by_row:
        row_valid.append(len(h & surviving_cols) >= min_cols)

    # Find maximal runs of consecutive valid rows that share the same
    # column signature (within tolerance).
    candidates: list[dict] = []
    i = 0
    n = len(rows)
    while i < n:
        if not row_valid[i]:
            i += 1
            continue
        sig = hits_by_row[i] & surviving_cols
        run_start = i
        run_end = i
        while run_end + 1 < n and row_valid[run_end + 1] \
                and len(hits_by_row[run_end + 1] & sig) >= min_cols:
            run_end += 1
            sig = sig | (hits_by_row[run_end] & surviving_cols)
        if run_end - run_start + 1 >= min_rows and len(sig) >= min_cols:
            run_rows = rows[run_start:run_end + 1]
            cell_items: list[dict] = []
            for row in run_rows:
                row_xs = _row_columns(row)
                # Keep only items that snap to a surviving column.
                for r in row:
                    x1 = int(r["x1"])
                    for j in sig:
                        if abs(x1 - centers[j]) <= col_tolerance:
                            cell_items.append(r)
                            break
            if not cell_items:
                i = run_end + 1
                continue
            # Uniformity gate: a real table has columns of roughly the
            # same width and rows of roughly the same height. Scattered
            # text that happens to align has wildly varying column widths
            # because the texts in each "column" are different categories
            # of content.
            sorted_sig = sorted(sig)
            col_centers_used = [centers[j] for j in sorted_sig]
            # Approximate column widths from gaps between successive
            # column centers — the last column extends to the right
            # edge of the matched cells.
            col_widths: list[float] = []
            for k in range(len(col_centers_used) - 1):
                col_widths.append(
                    col_centers_used[k + 1] - col_centers_used[k]
                )
            if not col_widths:
                # Single-column grid — degenerate, skip.
                i = run_end + 1
                continue
            row_heights = [
                max(int(r["y2"]) - int(r["y1"]) for r in row)
                for row in run_rows
                if row
            ]
            cv_cols = _coeff_var(col_widths)
            cv_rows = _coeff_var(row_heights)
            if cv_cols > max_col_width_cv or cv_rows > max_row_height_cv:
                i = run_end + 1
                continue
            # Density gate: at least min_density fraction of the
            # rows*cols grid must be populated. Sparse alignment is a
            # tell-tale sign of a non-table layout.
            density = len(cell_items) / float(
                max(1, (run_end - run_start + 1) * len(sig))
            )
            if density < min_density:
                i = run_end + 1
                continue
            x1 = min(int(r["x1"]) for r in cell_items) - padding
            y1 = min(int(r["y1"]) for r in cell_items) - padding
            x2 = max(int(r["x2"]) for r in cell_items) + padding
            y2 = max(int(r["y2"]) for r in cell_items) + padding
            candidates.append({
                "bbox": [max(0, x1), max(0, y1), int(x2), int(y2)],
                "rows": run_end - run_start + 1,
                "cols": len(sig),
                "cells_used": len(cell_items),
                "density": round(density, 3),
                "cv_cols": round(cv_cols, 3),
                "cv_rows": round(cv_rows, 3),
            })
        i = run_end + 1
    return candidates


def main() -> int:
    args = parse_args()
    ocr_path = Path(args.ocr)
    if not ocr_path.exists():
        sys.stderr.write(f"ERROR: OCR not found: {ocr_path}\n")
        return 1
    items = json.loads(ocr_path.read_text(encoding="utf-8"))
    candidates = detect(
        items,
        min_rows=args.min_rows,
        min_cols=args.min_cols,
        col_tolerance=args.col_tolerance_px,
        padding=args.padding_px,
    )
    out_json = json.dumps({"candidates": candidates},
                          ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_json, encoding="utf-8")
        sys.stderr.write(
            f"[detect_tables] {len(candidates)} candidate(s) → {args.out}\n"
        )
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
