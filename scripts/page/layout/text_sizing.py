"""Font-size ladder + per-run / per-line sizing rules.

Three things live here:

* `_FONT_LADDER` + `_snap_to_ladder` — the standard ladder OCR-derived
  sizes are snapped to, so two visually-identical labels do not render
  at different sizes because their bbox heights differ by 1-2 px.
* `_inherit_punct_sizes` + `_unify_run_sizes_by_color` — clean up the
  per-word sizing noise in mixed-class lines (e.g. `7分钟`).
* `font_size_pt` — the line-level formula `bbox_h × pt_per_px × α`,
  plus the stat-value heuristics that decide whether per-run sizing is
  safe to apply.
"""
from __future__ import annotations

import unicodedata
from collections import Counter


_FONT_LADDER = [8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 28, 32,
                36, 40, 44, 48, 54, 60, 66, 72, 80, 88]


# Resolution-independent calibration constants. These are pure ratios —
# they describe how PaddleOCR's bbox relates to the underlying glyph
# extent, independent of source resolution or slide size.
#
# α (CAPTURE_FACTOR_*) — bbox height in pt × α ≈ font pt. CJK glyphs
#   fill ~85 % of the em-box, and PaddleOCR pads the bbox by another
#   ~10 % vertically; the inverse of that combined factor is α ≈ 0.77.
#   Stat values are denser (digit-heavy) — OCR captures tighter, so α
#   is larger.
CAPTURE_FACTOR_SHORT = 0.78
CAPTURE_FACTOR_MEDIUM = 0.74
CAPTURE_FACTOR_LONG = 0.71


def _snap_to_ladder(pt: float) -> int:
    """Snap a font-size estimate to the nearest standard size."""
    return int(min(_FONT_LADDER, key=lambda v: abs(v - pt)))


def _is_punct_run(text: str) -> bool:
    """True when every visible char is Unicode P* (punctuation) or S*
    (symbol). Catches both ASCII (`+`, `"`, `,`) and CJK (`，`, `"`,
    `（`, `）`) without an explicit character list."""
    visible = "".join(c for c in (text or "") if not c.isspace())
    if not visible:
        return True
    return all(unicodedata.category(c)[0] in ("P", "S") for c in visible)


def _inherit_punct_sizes(runs: list[dict]) -> None:
    """Make punctuation runs follow neighbouring text's size.

    Mode C samples each PaddleOCR word_box independently. For
    punctuation like `+`, `"`, `，` the glyph itself is small (raw
    glyph_h ~8 px) so the per-word formula returns 8 pt. Typographically
    though punctuation shares the em-box with the surrounding text.

    Rule: for each pure-punctuation run, copy the size of the nearest
    non-punctuation neighbour (prefer the larger only on a tie).
    """
    if not runs:
        return
    is_punct = [_is_punct_run(r.get("text", "")) for r in runs]
    if not any(is_punct):
        return
    for i, punct in enumerate(is_punct):
        if not punct:
            continue
        prev_size = None
        prev_dist = None
        for j in range(i - 1, -1, -1):
            if not is_punct[j] and runs[j].get("size") is not None:
                prev_size = int(runs[j]["size"])
                prev_dist = i - j
                break
        next_size = None
        next_dist = None
        for j in range(i + 1, len(runs)):
            if not is_punct[j] and runs[j].get("size") is not None:
                next_size = int(runs[j]["size"])
                next_dist = j - i
                break
        if prev_size is None and next_size is None:
            continue
        if prev_size is None:
            runs[i]["size"] = next_size
        elif next_size is None:
            runs[i]["size"] = prev_size
        elif prev_dist == next_dist:
            runs[i]["size"] = max(prev_size, next_size)
        elif prev_dist < next_dist:
            runs[i]["size"] = prev_size
        else:
            runs[i]["size"] = next_size


def _unify_run_sizes_by_color(runs: list[dict]) -> None:
    """Snap same-colour runs within a record to their dominant size.

    Mode C's per-word glyph_h measurement can vary 1-2 ladder steps
    between adjacent CJK characters of the same visual size. Group runs
    by colour, take the mode size in each group, and snap any run
    within ±2 ladder steps to it. Outliers stay at their own size.
    """
    if not runs or len(runs) < 2:
        return
    by_color: dict[str, list[int]] = {}
    for i, r in enumerate(runs):
        if r.get("size") is None:
            continue
        by_color.setdefault(r["color"], []).append(i)
    for indices in by_color.values():
        if len(indices) < 2:
            continue
        sizes = [int(runs[i]["size"]) for i in indices]
        ladder_idx = {pt: i for i, pt in enumerate(_FONT_LADDER)}
        counter = Counter(sizes)
        max_count = max(counter.values())
        candidates = [s for s, c in counter.items() if c == max_count]
        dominant = max(candidates)
        d_idx = ladder_idx.get(dominant, -1)
        if d_idx < 0:
            continue
        for i in indices:
            cur = int(runs[i]["size"])
            cur_idx = ladder_idx.get(cur, -1)
            if cur_idx < 0:
                continue
            if abs(cur_idx - d_idx) <= 2:
                runs[i]["size"] = dominant


def font_size_pt(text: str, bbox_w: int, bbox_h: int,
                 pt_per_px: float = 0.75) -> int:
    """Derive font size from bbox HEIGHT only (width ignored).

    Formula:  font_pt = bbox_h_px × pt_per_px × α
    where pt_per_px = slide_h_in × 72 / source_h_px and α reflects how
    tightly the OCR bbox hugs the glyph extent.

    Width is unreliable for mixed-glyph stats because PaddleOCR
    occasionally varies the per-glyph width estimate between visually
    identical lines. Height is the stable axis; cross-record drift gets
    cleaned up at PPTX-build time in apply_size_unification (ratio cap).
    """
    if len(text) <= 4:
        alpha = CAPTURE_FACTOR_SHORT
    elif len(text) <= 12:
        alpha = CAPTURE_FACTOR_MEDIUM
    else:
        alpha = CAPTURE_FACTOR_LONG
    raw = bbox_h * pt_per_px * alpha
    # Two-tier cap: content text capped at 36 pt because OCR bboxes pad
    # heavily around decorated stat numbers; cover/hero titles can be
    # genuinely huge. ≤ 60 pt → cap 36; > 60 pt → up to ladder max.
    if raw > 60.0:
        clamped = min(88.0, raw)
    else:
        clamped = min(36.0, raw)
    return _snap_to_ladder(max(8.0, clamped))


def _is_stat_value(text: str) -> bool:
    """Stat-value heuristic: short text with digits and a unit suffix.
    Matches `1000亿元`, `24.22%`, `63个`, `29 PB`, etc."""
    compact = "".join(c for c in (text or "") if not c.isspace())
    if not any(c.isdigit() for c in compact) or len(compact) > 8:
        return False
    units = [
        "亿元", "PB", "MB", "GB", "TB",
        "分钟", "小时",
        "亿", "%", "个", "万", "轮", "条", "次", "倍", "年", "月", "天",
    ]
    for unit in units:
        if not compact.endswith(unit):
            continue
        number = compact[:-len(unit)] if unit else compact
        return bool(number) and all(c.isdigit() or c in ".,+-" for c in number)
    return False


def _should_apply_run_font_sizes(text: str) -> bool:
    """Only short numeric stats get per-run font sizes.

    Per-word glyph height is useful for `524个` or `7分钟`. It is
    harmful for headings and formulas (`Kona·核心洞察`, `O(N)→O(log N)`):
    tiny punctuation or Latin word boxes shrink individual runs and
    create visible drift. Those lines keep one element-level size.
    """
    compact = "".join(c for c in (text or "") if not c.isspace())
    if not _is_stat_value(compact):
        return False
    formula_chars = set("()[]{}=<>→←+-*/")
    latin = sum(1 for c in compact if c.isascii() and c.isalpha())
    if any(c in formula_chars for c in compact) and latin >= 2:
        return False
    return True
