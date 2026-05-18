"""HTML report generation for one regression run.

Per fixture we emit:
  - source.png, baseline.png, current.png (copies for the static report)
  - side_by_side.png    horizontal triptych for at-a-glance review
  - diff_vs_source.png  pixel-diff heatmap, source vs current
  - diff_vs_baseline.png  pixel-diff heatmap, baseline vs current
  - worst_regions.png   source image with red boxes around top regions

And a single report.html that links them all together with scores + deltas.
"""
from __future__ import annotations

import html
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tests.runner.baseline import Comparison, Status
from tests.runner.metrics import FixtureScores, COMPARE_W, COMPARE_H, _load_for_compare

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = REPO_ROOT / "tests" / "reports"

WORST_WINDOW = 160              # px window for region scoring
WORST_STRIDE = 80
WORST_N = 3                     # top regions to report


# ---------------------------------------------------------------------- diff imagery

def _heatmap(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """Cool→hot colormap on the absolute pixel difference (per-channel mean)."""
    diff = np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16)).mean(axis=2)
    # Stretch for visibility — caps at ~80 to keep typical anti-aliasing dim
    norm = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_image(arr: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(arr).copy()
    d = ImageDraw.Draw(img)
    pad = 14
    d.rectangle([0, 0, 360, 56], fill=(0, 0, 0))
    d.text((pad, 12), label, fill=(255, 255, 255), font=_font(28))
    return np.asarray(img)


def _triptych(src: np.ndarray, base: np.ndarray | None, curr: np.ndarray) -> np.ndarray:
    """Horizontal source | baseline | current strip."""
    base = base if base is not None else np.full_like(src, 230)
    panels = [
        _label_image(src, "SOURCE"),
        _label_image(base, "BASELINE (prev)"),
        _label_image(curr, "CURRENT"),
    ]
    return np.hstack(panels)


# ---------------------------------------------------------------------- worst regions

@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    score: float            # mean abs diff inside the window
    cause_guess: str


def _guess_cause(window_src: np.ndarray, window_cur: np.ndarray) -> str:
    """Crude heuristic: what feature changed most inside this region?"""
    src_gray = cv2.cvtColor(window_src, cv2.COLOR_RGB2GRAY)
    cur_gray = cv2.cvtColor(window_cur, cv2.COLOR_RGB2GRAY)

    # Edge density delta — proxy for "text density" or "shape boundary" change
    src_edges = cv2.Canny(src_gray, 80, 160).mean()
    cur_edges = cv2.Canny(cur_gray, 80, 160).mean()
    edge_delta = abs(src_edges - cur_edges)

    # Mean color delta — proxy for color/fill rendering issues
    color_delta = float(np.abs(
        window_src.mean(axis=(0, 1)) - window_cur.mean(axis=(0, 1))
    ).mean())

    # Brightness std delta — proxy for content presence vs flat fill
    src_std = float(src_gray.std())
    cur_std = float(cur_gray.std())
    std_delta = abs(src_std - cur_std)

    reasons = []
    if edge_delta > 4 and cur_edges < src_edges:
        reasons.append("文字或图标可能缺失/被压平（边缘密度下降）")
    elif edge_delta > 4 and cur_edges > src_edges:
        reasons.append("可能多出了不在源图里的边缘（多余对象或字号偏大）")
    if color_delta > 25:
        reasons.append(f"主色调偏差较大（ΔRGB ≈ {color_delta:.0f}）")
    if std_delta > 25 and cur_std < src_std:
        reasons.append("区域被填成接近纯色，可能丢失了内容")
    if not reasons:
        if edge_delta > 1.5:
            reasons.append("轻微的位置/字号偏移")
        else:
            reasons.append("整体细节差异（可能是抗锯齿/渲染差）")
    return "；".join(reasons) + " ⚠ 启发式猜测"


def _worst_regions(src: np.ndarray, curr: np.ndarray) -> list[Region]:
    diff = np.abs(src.astype(np.int16) - curr.astype(np.int16)).mean(axis=2)
    h, w = diff.shape

    candidates: list[Region] = []
    for y in range(0, max(1, h - WORST_WINDOW + 1), WORST_STRIDE):
        for x in range(0, max(1, w - WORST_WINDOW + 1), WORST_STRIDE):
            window = diff[y:y + WORST_WINDOW, x:x + WORST_WINDOW]
            score = float(window.mean())
            if score < 5.0:    # discard near-clean windows
                continue
            cause = _guess_cause(
                src[y:y + WORST_WINDOW, x:x + WORST_WINDOW],
                curr[y:y + WORST_WINDOW, x:x + WORST_WINDOW],
            )
            candidates.append(Region(x, y, WORST_WINDOW, WORST_WINDOW, score, cause))

    # Greedy NMS — discard windows that overlap an already-picked higher-scored window
    candidates.sort(key=lambda r: r.score, reverse=True)
    chosen: list[Region] = []
    for c in candidates:
        keep = True
        for k in chosen:
            if not (c.x + c.w <= k.x or k.x + k.w <= c.x or
                    c.y + c.h <= k.y or k.y + k.h <= c.y):
                keep = False
                break
        if keep:
            chosen.append(c)
        if len(chosen) >= WORST_N:
            break
    return chosen


def _annotate_regions(src: np.ndarray, regions: list[Region]) -> np.ndarray:
    img = Image.fromarray(src).copy()
    d = ImageDraw.Draw(img)
    font = _font(36)
    for i, r in enumerate(regions, 1):
        d.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=(255, 0, 0), width=6)
        label_w, label_h = 56, 56
        lx = r.x
        ly = max(0, r.y - label_h)
        d.rectangle([lx, ly, lx + label_w, ly + label_h], fill=(255, 0, 0))
        d.text((lx + 14, ly + 6), str(i), fill=(255, 255, 255), font=font)
    return np.asarray(img)


# ---------------------------------------------------------------------- HTML

def _fmt(val):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _status_class(status: Status) -> str:
    return {
        Status.NEW: "status-new",
        Status.UNCHANGED: "status-unchanged",
        Status.IMPROVED: "status-improved",
        Status.REGRESSED: "status-regressed",
        Status.FAILED: "status-failed",
    }[status]


def _status_glyph(status: Status) -> str:
    return {
        Status.NEW: "NEW",
        Status.UNCHANGED: "=",
        Status.IMPROVED: "▲",
        Status.REGRESSED: "▼",
        Status.FAILED: "✗",
    }[status]


def _fixture_section(scores: FixtureScores, comp: Comparison,
                     paths: dict, regions: list[Region]) -> str:
    rows = []
    for key, (prev, now, delta) in comp.deltas.items():
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        cls = "delta-up" if delta > 0 else ("delta-down" if delta < 0 else "")
        rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{prev:.4f}</td>"
            f"<td>{now:.4f}</td>"
            f"<td class='{cls}'>{arrow} {delta:+.4f}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4'>(no baseline)</td></tr>")

    regions_html = ""
    if regions:
        items = "".join(
            f"<li><b>区域 {i}</b> @ ({r.x},{r.y}) — {html.escape(r.cause_guess)}"
            f" (mean diff {r.score:.1f})</li>"
            for i, r in enumerate(regions, 1)
        )
        regions_html = (
            f"<h4>Worst regions</h4>"
            f"<img src='{paths['worst']}' class='diff-img'>"
            f"<ul class='regions'>{items}</ul>"
        )

    missing = scores.text.get("missing_keywords") or []
    missing_html = ""
    if missing:
        missing_html = (
            "<h4>Missing keywords</h4>"
            f"<ul class='missing'>{''.join(f'<li>{html.escape(m)}</li>' for m in missing)}</ul>"
        )

    reasons = comp.regression_reasons + comp.improvement_notes
    reasons_html = ""
    if reasons:
        reasons_html = "<ul>" + "".join(
            f"<li>{html.escape(r)}</li>" for r in reasons
        ) + "</ul>"

    return f"""\
<section id='{scores.fixture}' class='fixture {_status_class(comp.status)}'>
  <h2>
    <span class='glyph'>{_status_glyph(comp.status)}</span>
    {html.escape(scores.fixture)}
    <span class='composite'>composite {scores.composite:.4f}</span>
  </h2>
  {reasons_html}
  <div class='triptych'>
    <figure><img src='{paths['source']}'><figcaption>SOURCE (truth)</figcaption></figure>
    <figure><img src='{paths['baseline']}'><figcaption>BASELINE (prev)</figcaption></figure>
    <figure><img src='{paths['current']}'><figcaption>CURRENT</figcaption></figure>
  </div>
  <div class='diffs'>
    <figure><img src='{paths['diff_src']}'><figcaption>diff vs SOURCE (绝对质量)</figcaption></figure>
    <figure><img src='{paths['diff_base']}'><figcaption>diff vs BASELINE (本次变化)</figcaption></figure>
  </div>
  {regions_html}
  <h4>Metric deltas vs baseline</h4>
  <table class='deltas'>
    <thead><tr><th>metric</th><th>prev</th><th>now</th><th>Δ</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {missing_html}
  <details>
    <summary>Raw scores</summary>
    <pre>{html.escape(json.dumps(scores.to_dict(), ensure_ascii=False, indent=2))}</pre>
  </details>
</section>
"""


# ---------------------------------------------------------------------- public API

def render_report(run_results: list[tuple[FixtureScores, Comparison,
                                          Path, Path, Path | None]],
                  out_root: Path | None = None) -> Path:
    """Render the full HTML report.

    run_results: list of (scores, comparison, source_png, current_png, baseline_png_or_None).
    Returns the path to report.html.
    """
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    out_root = out_root or REPORTS_ROOT / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    fx_root = out_root / "fixtures"
    fx_root.mkdir(parents=True, exist_ok=True)

    sections = []
    summary_rows = []
    composite_now_total = 0.0
    composite_prev_total = 0.0
    composite_prev_count = 0
    regressed = []
    improved = []

    for scores, comp, src_path, curr_path, base_path in run_results:
        fx_dir = fx_root / scores.fixture
        fx_dir.mkdir(parents=True, exist_ok=True)

        # Copy original images
        shutil.copy2(src_path, fx_dir / "source.png")
        shutil.copy2(curr_path, fx_dir / "current.png")
        if base_path and base_path.exists():
            shutil.copy2(base_path, fx_dir / "baseline.png")
            base_arr = _load_for_compare(base_path)
        else:
            base_arr = None

        # Compute diff imagery on canonical size
        src_arr = _load_for_compare(src_path)
        curr_arr = _load_for_compare(curr_path)

        diff_src = _heatmap(src_arr, curr_arr)
        _save(diff_src, fx_dir / "diff_vs_source.png")

        if base_arr is not None:
            diff_base = _heatmap(base_arr, curr_arr)
        else:
            diff_base = np.full_like(src_arr, 230)
        _save(diff_base, fx_dir / "diff_vs_baseline.png")

        regions = _worst_regions(src_arr, curr_arr)
        annotated = _annotate_regions(src_arr, regions)
        _save(annotated, fx_dir / "worst_regions.png")

        rel = lambda name: f"fixtures/{scores.fixture}/{name}"
        paths = {
            "source": rel("source.png"),
            "baseline": rel("baseline.png") if base_path and base_path.exists() else rel("source.png"),
            "current": rel("current.png"),
            "diff_src": rel("diff_vs_source.png"),
            "diff_base": rel("diff_vs_baseline.png"),
            "worst": rel("worst_regions.png"),
        }
        sections.append(_fixture_section(scores, comp, paths, regions))

        # Summary row
        prev_disp = f"{comp.composite_prev:.4f}" if comp.composite_prev is not None else "—"
        summary_rows.append(
            f"<tr class='{_status_class(comp.status)}'>"
            f"<td><a href='#{scores.fixture}'>{html.escape(scores.fixture)}</a></td>"
            f"<td>{_status_glyph(comp.status)}</td>"
            f"<td>{prev_disp}</td>"
            f"<td>{scores.composite:.4f}</td>"
            f"<td>{scores.runtime_seconds:.1f}s {'(cached)' if scores.cached_pipeline else ''}</td>"
            f"</tr>"
        )
        composite_now_total += scores.composite
        if comp.composite_prev is not None:
            composite_prev_total += comp.composite_prev
            composite_prev_count += 1
        if comp.status == Status.REGRESSED:
            regressed.append(scores.fixture)
        elif comp.status == Status.IMPROVED:
            improved.append(scores.fixture)

    avg_now = composite_now_total / max(len(run_results), 1)
    if composite_prev_count:
        avg_prev = composite_prev_total / composite_prev_count
        avg_delta = f"{avg_now - avg_prev:+.4f}"
    else:
        avg_delta = "—"

    header_status = (
        f"<b>{len(regressed)} regression(s)</b>" if regressed
        else f"<b>{len(improved)} improvement(s)</b>" if improved
        else "no significant changes"
    )

    html_out = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'>
<title>DeckWeaver Test Report — {timestamp}</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", sans-serif;
         margin: 24px; color: #1F2937; }}
  h1 {{ margin-bottom: 4px; }}
  .header-meta {{ color: #6B7280; margin-bottom: 24px; }}
  table.summary {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
  table.summary th, table.summary td {{ padding: 6px 10px; border-bottom: 1px solid #E5E7EB;
                                        text-align: left; }}
  table.summary tr.status-regressed td {{ background: #FEF2F2; }}
  table.summary tr.status-improved td {{ background: #ECFDF5; }}
  table.summary tr.status-failed td {{ background: #FECACA; }}
  table.summary tr.status-new td {{ background: #EFF6FF; }}

  section.fixture {{ border: 1px solid #E5E7EB; border-radius: 12px;
                     padding: 16px 20px; margin: 24px 0; }}
  section.fixture.status-regressed {{ border-color: #FCA5A5; background: #FEF2F2; }}
  section.fixture.status-improved {{ border-color: #6EE7B7; background: #ECFDF5; }}
  section.fixture.status-failed {{ border-color: #DC2626; background: #FECACA; }}
  section.fixture.status-new {{ border-color: #93C5FD; background: #EFF6FF; }}

  .glyph {{ display:inline-block; min-width: 28px; }}
  .composite {{ font-weight: 400; color: #6B7280; margin-left: 12px; font-size: 0.9em; }}
  .triptych, .diffs {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;
                       margin: 12px 0; }}
  .diffs {{ grid-template-columns: 1fr 1fr; }}
  .triptych figure, .diffs figure {{ margin: 0; }}
  .triptych img, .diffs img, .diff-img {{ width: 100%; border: 1px solid #D1D5DB;
                                            border-radius: 6px; display: block; }}
  figcaption {{ font-size: 0.85em; color: #6B7280; text-align: center; margin-top: 4px; }}
  table.deltas {{ border-collapse: collapse; margin-top: 8px; }}
  table.deltas th, table.deltas td {{ padding: 4px 10px; border-bottom: 1px solid #E5E7EB;
                                      text-align: left; font-variant-numeric: tabular-nums; }}
  td.delta-down {{ color: #B91C1C; font-weight: 600; }}
  td.delta-up {{ color: #047857; font-weight: 600; }}
  ul.regions li, ul.missing li {{ margin: 4px 0; }}
  details {{ margin-top: 16px; }}
  pre {{ background: #F3F4F6; padding: 8px; border-radius: 6px; overflow-x: auto; }}
</style>
</head><body>
<h1>DeckWeaver Test Report</h1>
<div class='header-meta'>
  {timestamp} · average composite {avg_now:.4f} (Δ {avg_delta}) · {header_status}
</div>

<h2>Summary</h2>
<table class='summary'>
  <thead><tr><th>fixture</th><th>status</th><th>prev composite</th>
             <th>now composite</th><th>runtime</th></tr></thead>
  <tbody>{''.join(summary_rows)}</tbody>
</table>

{''.join(sections)}

</body></html>
"""
    report_path = out_root / "report.html"
    report_path.write_text(html_out, encoding="utf-8")
    return report_path
