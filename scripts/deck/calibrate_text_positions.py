#!/usr/bin/env python
"""Closed-loop text-position calibration from rendered PPT previews.

Builds a temporary text-only deck where every text element is painted with
a unique colour. After LibreOffice renders that calibration deck, each
text element's actual rendered ink bbox is detected by colour and compared
with the source image's target ink bbox. The measured dx/dy is written
back into the layout.
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "page"))
sys.path.insert(0, str(SCRIPTS_ROOT / "deck"))
sys.path.insert(0, str(SCRIPTS_ROOT / "verify"))

from image_sources import SUPPORTED_IMAGE_EXTENSIONS, supported_image_formats  # noqa: E402
from inventory_to_layout import (  # noqa: E402
    _sample_background_for_bbox,
    _target_text_metrics,
)
import build_pptx_from_layout as _builder  # noqa: E402
import render_preview as _render  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layout", required=True, help="Combined layout JSON.")
    p.add_argument("--source-dir", required=True,
                   help=f"Directory with page_NN source images "
                        f"({supported_image_formats()}).")
    p.add_argument("--work-dir", required=True,
                   help="Working directory for calibration artifacts.")
    p.add_argument("--assets-root", default=None,
                   help="Assets root for build_pptx_from_layout.")
    p.add_argument("--out-layout", default=None,
                   help="Output layout path. Default: update --layout in place.")
    p.add_argument("--iterations", type=int, default=1,
                   help="Calibration iterations (default: 1).")
    p.add_argument("--dpi", type=int, default=100,
                   help="Calibration preview DPI (default: 100).")
    p.add_argument("--max-shift", type=float, default=30.0,
                   help="Maximum absolute dx/dy in source px per iteration.")
    return p.parse_args()


_CALIBRATION_PALETTE: tuple[tuple[int, int, int], ...] = (
    (190, 0, 0),
    (0, 118, 0),
    (0, 48, 210),
    (190, 0, 165),
    (0, 145, 170),
    (220, 105, 0),
    (95, 0, 210),
    (0, 0, 0),
    (128, 64, 0),
    (0, 96, 120),
    (170, 0, 80),
    (55, 110, 0),
)


def _unique_colour(slide_idx: int, text_idx: int) -> tuple[int, int, int]:
    # Fallback only. Normal calibration uses graph-coloured palette
    # assignments so nearby text boxes have strongly separated colours.
    h = (0.137 + (slide_idx * 0.173) + (text_idx * 0.61803398875)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.92, 0.50)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _hex(colour: tuple[int, int, int]) -> str:
    r, g, b = colour
    return f"#{r:02X}{g:02X}{b:02X}"


def _text_elements(slide: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        e for e in slide.get("elements", [])
        if (e.get("type") or "").lower() == "text"
        and str(e.get("text", "") or "").strip()
        and e.get("box") and len(e["box"]) == 4
    ]


def _expanded_text_box(el: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in el.get("box", [0, 0, 0, 0]))
    mx = max(36.0, min(180.0, w * 0.50))
    my = max(36.0, min(150.0, h * 3.50))
    return x - mx, y - my, x + w + mx, y + h + my


def _boxes_overlap(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _assign_calibration_colours(
    texts: list[dict[str, Any]],
    slide_idx: int,
) -> list[tuple[int, int, int]]:
    """Reuse a small high-contrast palette without local collisions.

    The detector searches in an expanded neighbourhood around each text
    box. If two nearby texts have similar hue, one detector can mistakenly
    union both ink bboxes. This greedy graph-colouring step ensures
    overlapping search neighbourhoods do not share a palette colour, while
    allowing distant texts to reuse colours safely.
    """
    expanded = [_expanded_text_box(el) for el in texts]
    assigned: list[tuple[int, int, int] | None] = [None] * len(texts)
    order = sorted(
        range(len(texts)),
        key=lambda i: (
            -sum(1 for j in range(len(texts))
                 if i != j and _boxes_overlap(expanded[i], expanded[j])),
            expanded[i][1],
            expanded[i][0],
        ),
    )
    for idx in order:
        used = {
            assigned[j] for j in range(len(texts))
            if assigned[j] is not None and _boxes_overlap(expanded[idx], expanded[j])
        }
        for colour in _CALIBRATION_PALETTE:
            if colour not in used:
                assigned[idx] = colour
                break
        if assigned[idx] is None:
            assigned[idx] = _unique_colour(slide_idx, idx)
    return [c if c is not None else _unique_colour(slide_idx, i)
            for i, c in enumerate(assigned)]


def _calibration_layout(layout: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[int, int], tuple[int, int, int]]]:
    out = copy.deepcopy(layout)
    colours: dict[tuple[int, int], tuple[int, int, int]] = {}
    slides = out.get("slides") or [out]
    for s_idx, slide in enumerate(slides):
        text_only = []
        texts = _text_elements(slide)
        assigned_colours = _assign_calibration_colours(texts, s_idx)
        for t_idx, el in enumerate(texts):
            colour = assigned_colours[t_idx]
            colours[(s_idx, t_idx)] = colour
            c = copy.deepcopy(el)
            c["color"] = _hex(colour)
            c["fill"] = "transparent"
            c["line"] = "transparent"
            if c.get("runs"):
                for run in c["runs"]:
                    run["color"] = _hex(colour)
            text_only.append(c)
        slide["background"] = "#FFFFFF"
        slide["elements"] = text_only
    return out, colours


def _source_for_slide(slide: dict[str, Any], source_dir: Path,
                      slide_idx: int) -> Path | None:
    name = str(slide.get("name") or "")
    match = re.search(r"page[_-]?(\d+)", name)
    candidates: list[Path] = []
    if match:
        n = int(match.group(1))
        candidates.extend(sorted(source_dir.glob(f"page_{n:02d}.*")))
        candidates.extend(sorted(source_dir.glob(f"page_{n}.*")))
    candidates.extend(sorted(source_dir.glob(f"page_{slide_idx + 1:02d}.*")))
    candidates.extend(sorted(source_dir.glob(f"page_{slide_idx + 1}.*")))
    exts = set(SUPPORTED_IMAGE_EXTENSIONS)
    for path in candidates:
        if path.suffix.lower() in exts and path.exists():
            return path
    return None


def _source_to_preview_transform(slide: dict[str, Any],
                                 preview_shape: tuple[int, int]) -> tuple[float, float, float]:
    ph, pw = preview_shape
    sw = float(slide.get("source_width") or 1280)
    sh = float(slide.get("source_height") or 720)
    scale = min(pw / sw, ph / sh)
    ox = (pw - sw * scale) / 2.0
    oy = (ph - sh * scale) / 2.0
    return scale, ox, oy


def _box_to_preview(box: list[float], scale: float,
                    ox: float, oy: float) -> tuple[int, int, int, int]:
    x, y, w, h = (float(v) for v in box)
    x1 = int(np.floor(ox + x * scale))
    y1 = int(np.floor(oy + y * scale))
    x2 = int(np.ceil(ox + (x + w) * scale))
    y2 = int(np.ceil(oy + (y + h) * scale))
    return x1, y1, x2, y2


def _detect_colour_bbox(
    preview_rgb: np.ndarray,
    colour: tuple[int, int, int],
    search_box: tuple[int, int, int, int],
    expected_box: tuple[int, int, int, int] | None = None,
    competing_colours: list[tuple[int, int, int]] | None = None,
) -> tuple[int, int, int, int] | None:
    h, w = preview_rgb.shape[:2]
    x1, y1, x2, y2 = search_box
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = preview_rgb[y1:y2, x1:x2].astype(np.float32)
    bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    target = np.array(colour, dtype=np.float32)
    vec = target - bg
    denom = float(np.dot(vec, vec))
    if denom <= 0:
        return None
    rel = crop - bg
    alpha = np.tensordot(rel, vec, axes=([2], [0])) / denom
    recon = bg + alpha[:, :, None] * vec
    residual = np.linalg.norm(crop - recon, axis=2)
    near_white = np.max(np.abs(crop - bg), axis=2) < 8
    mask = (alpha > 0.10) & (alpha < 1.25) & (residual < 45) & ~near_white
    if competing_colours:
        competitors = [c for c in competing_colours if tuple(c) != tuple(colour)]
        if competitors:
            comp = np.array(competitors, dtype=np.float32)
            comp_vecs = comp - bg
            comp_den = np.einsum("ij,ij->i", comp_vecs, comp_vecs)
            valid = comp_den > 0
            if np.any(valid):
                comp_vecs = comp_vecs[valid]
                comp_den = comp_den[valid]
                comp_alpha = np.tensordot(rel, comp_vecs, axes=([2], [1])) / comp_den
                comp_recon = bg + comp_alpha[:, :, :, None] * comp_vecs[None, None, :, :]
                comp_residual = np.linalg.norm(crop[:, :, None, :] - comp_recon, axis=3)
                best_competing = np.min(comp_residual, axis=2)
                # Keep pixels that clearly fit this colour line better
                # than any other colour used in the same search region.
                mask &= residual <= (best_competing - 6.0)
    if int(mask.sum()) < 3:
        return None
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((2, 2), np.uint8))
    if expected_box is not None:
        # Long titles are often horizontally disconnected character runs,
        # so a connected-component union can accidentally keep only the
        # middle of the line. Select the row band closest to this text
        # box, then keep all candidate pixels on that band.
        ex1, ey1, ex2, ey2 = expected_box
        expected_cy = ((ey1 + ey2) / 2.0) - y1
        rows = (mask.sum(axis=1) > 0).astype(np.uint8)[:, None]
        rows = cv2.dilate(rows, np.ones((7, 1), np.uint8)).ravel() > 0
        groups: list[tuple[int, int]] = []
        start: int | None = None
        for idx, on in enumerate(rows.tolist() + [False]):
            if on and start is None:
                start = idx
            elif not on and start is not None:
                groups.append((start, idx))
                start = None
        if groups:
            best = min(
                groups,
                key=lambda g: (
                    abs(((g[0] + g[1]) / 2.0) - expected_cy),
                    max(0, (g[1] - g[0]) - max(24, (ey2 - ey1) * 3)),
                ),
            )
            selected_rows = np.zeros(mask.shape[0], dtype=bool)
            selected_rows[best[0]:best[1]] = True
            selected = mask & selected_rows[:, None].astype(np.uint8)
            if int(selected.sum()) >= 3:
                mask = selected
        ex1, ey1, ex2, ey2 = expected_box
        pad_y = int(round(max(4, min(10, (ey2 - ey1) * 0.35))))
        row_start = max(0, (ey1 - y1) - pad_y)
        row_end = min(mask.shape[0], (ey2 - y1) + pad_y)
        if row_end > row_start:
            selected_rows = np.zeros(mask.shape[0], dtype=bool)
            selected_rows[row_start:row_end] = True
            selected = mask & selected_rows[:, None].astype(np.uint8)
            if int(selected.sum()) >= 3:
                mask = selected
        pad_x = int(round(max(6, min(16, (ex2 - ex1) * 0.20))))
        col_start = max(0, (ex1 - x1) - pad_x)
        col_end = min(mask.shape[1], (ex2 - x1) + pad_x)
        if col_end > col_start:
            selected_cols = np.zeros(mask.shape[1], dtype=bool)
            selected_cols[col_start:col_end] = True
            selected = mask & selected_cols[None, :].astype(np.uint8)
            if int(selected.sum()) >= 3:
                mask = selected
    ys, xs = np.where(mask > 0)
    if xs.size < 2 or ys.size < 2:
        return None
    return (
        int(x1 + xs.min()),
        int(y1 + ys.min()),
        int(x1 + xs.max() + 1),
        int(y1 + ys.max() + 1),
    )


def _target_ink_for_element(el: dict[str, Any],
                            source_bgr: np.ndarray) -> list[int] | None:
    value = _target_ink_from_char_boxes(el, source_bgr)
    if value is not None:
        return value
    for key in ("target_ink", "fit_target_ink"):
        value = el.get(key)
        if value and len(value) == 4:
            return [int(v) for v in value]
    bbox = el.get("source_bbox")
    if bbox and len(bbox) == 4:
        metrics = _target_text_metrics(source_bgr, tuple(int(v) for v in bbox))
        if metrics is not None:
            return [int(v) for v in metrics["ink_bbox"]]
    return None


def _target_ink_from_char_boxes(el: dict[str, Any],
                                source_bgr: np.ndarray) -> list[int] | None:
    boxes = el.get("source_char_boxes") or el.get("char_boxes")
    if not boxes:
        return None
    text = str(el.get("text", "") or "")
    chars = el.get("source_chars") or list(text)
    if len(boxes) != len(chars):
        return None
    source_bbox = el.get("source_bbox")
    if source_bbox and len(source_bbox) == 4:
        line_bbox = tuple(int(v) for v in source_bbox)
    else:
        line_bbox = (
            min(int(b[0]) for b in boxes if len(b) == 4),
            min(int(b[1]) for b in boxes if len(b) == 4),
            max(int(b[2]) for b in boxes if len(b) == 4),
            max(int(b[3]) for b in boxes if len(b) == 4),
        )
    line_bg = _sample_background_for_bbox(source_bgr, line_bbox)

    inks: list[list[int]] = []
    for ch, box in zip(chars, boxes):
        if not str(ch).strip() or len(box) != 4:
            continue
        ink = _char_box_ink_bbox(source_bgr, box, line_bg)
        if ink is None:
            continue
        inks.append(ink)
    if not inks:
        return None
    return [
        min(b[0] for b in inks),
        min(b[1] for b in inks),
        max(b[2] for b in inks),
        max(b[3] for b in inks),
    ]


def _char_box_ink_bbox(
    source_bgr: np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    line_bg: np.ndarray,
) -> list[int] | None:
    h_img, w_img = source_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    sub = source_bgr[y1:y2, x1:x2]
    if sub.size == 0:
        return None

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1] > 50
    local_bg = line_bg
    if int(sat.sum()) > 0.45 * sat.size:
        local_bg = np.median(sub[sat], axis=0)
    diff = np.abs(sub.astype(int) - local_bg.astype(int)).max(axis=2)
    mask = diff > 35

    if int(sat.sum()) > 0.45 * sat.size:
        bg_hsv = cv2.cvtColor(
            np.array([[local_bg]], dtype=np.uint8),
            cv2.COLOR_BGR2HSV,
        )[0, 0]
        light = (
            (hsv[:, :, 1] < 130)
            & (hsv[:, :, 2].astype(int) > int(bg_hsv[2]) + 25)
        )
        if int(light.sum()) >= 2:
            mask = light

    if int(mask.sum()) < 2:
        return None
    row_has_ink = mask.sum(axis=1) >= 1
    col_has_ink = mask.sum(axis=0) >= 1
    rows = np.where(row_has_ink)[0]
    cols = np.where(col_has_ink)[0]
    if rows.size < 2 or cols.size < 1:
        return None
    return [
        x1 + int(cols.min()),
        y1 + int(rows.min()),
        x1 + int(cols.max()) + 1,
        y1 + int(rows.max()) + 1,
    ]


def _source_bbox(el: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = el.get("source_bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _is_dense_small_text(el: dict[str, Any],
                         slide: dict[str, Any]) -> bool:
    bbox = _source_bbox(el)
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    size = float(el.get("size") or el.get("font_size") or 0)
    visible = sum(1 for ch in str(el.get("text") or "") if not ch.isspace())
    if h > 24 or size > 13 or visible < 2:
        return False
    cx = (x1 + x2) / 2.0
    neighbours = 0
    for other in _text_elements(slide):
        if other is el:
            continue
        obox = _source_bbox(other)
        if obox is None:
            continue
        ox1, oy1, ox2, oy2 = obox
        oh = oy2 - oy1
        if oh > 26:
            continue
        ocx = (ox1 + ox2) / 2.0
        y_gap = max(0.0, max(y1, oy1) - min(y2, oy2))
        overlap = max(0.0, min(x2, ox2) - max(x1, ox1))
        narrower = max(1.0, min(x2 - x1, ox2 - ox1))
        same_column = (
            overlap / narrower >= 0.20
            or abs(cx - ocx) <= max(80.0, min(180.0, (x2 - x1) * 1.35))
        )
        if same_column and y_gap <= 18:
            neighbours += 1
            if neighbours >= 1:
                return True
    return False


def _apply_iteration(layout: dict[str, Any],
                     colours: dict[tuple[int, int], tuple[int, int, int]],
                     preview_dir: Path,
                     source_dir: Path,
                     max_shift: float,
                     iteration: int) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    slides = layout.get("slides") or [layout]
    for s_idx, slide in enumerate(slides):
        slide_colours = [
            c for (slide_no, _text_no), c in colours.items()
            if slide_no == s_idx
        ]
        preview_path = preview_dir / f"page-{s_idx + 1}.png"
        preview_bgr = cv2.imread(str(preview_path))
        source_path = _source_for_slide(slide, source_dir, s_idx)
        source_bgr = cv2.imread(str(source_path)) if source_path else None
        if preview_bgr is None or source_bgr is None:
            continue
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        scale, ox, oy = _source_to_preview_transform(slide, preview_rgb.shape[:2])
        for t_idx, el in enumerate(_text_elements(slide)):
            colour = colours.get((s_idx, t_idx))
            if colour is None:
                continue
            px1, py1, px2, py2 = _box_to_preview(el["box"], scale, ox, oy)
            box_w = max(1, px2 - px1)
            box_h = max(1, py2 - py1)
            mx = int(round(max(18, min(90, max(box_w * 0.18,
                                                max_shift * scale + 12)))))
            my = int(round(max(18, min(70, max(box_h * 2.0,
                                                max_shift * scale + 12)))))
            rendered = _detect_colour_bbox(
                preview_rgb, colour,
                (px1 - mx, py1 - my, px2 + mx, py2 + my),
                (px1, py1, px2, py2),
                competing_colours=slide_colours,
            )
            target = _target_ink_for_element(el, source_bgr)
            if rendered is None or target is None:
                report.append({
                    "slide": s_idx + 1,
                    "iteration": iteration,
                    "name": el.get("name"),
                    "text": el.get("text"),
                    "status": "skipped",
                })
                continue
            rx1 = (rendered[0] - ox) / scale
            ry1 = (rendered[1] - oy) / scale
            rx2 = (rendered[2] - ox) / scale
            ry2 = (rendered[3] - oy) / scale
            tx1, ty1, tx2, ty2 = (float(v) for v in target)
            if (el.get("align") or "").lower() == "center":
                dx = ((tx1 + tx2) / 2.0) - ((rx1 + rx2) / 2.0)
            else:
                dx = tx1 - rx1
            if _is_dense_small_text(el, slide):
                dy = ((ty1 + ty2) / 2.0) - ((ry1 + ry2) / 2.0)
            else:
                dy = ty1 - ry1
            dx = float(max(-max_shift, min(max_shift, dx)))
            dy = float(max(-max_shift, min(max_shift, dy)))
            if abs(dx) < 0.15:
                dx = 0.0
            if abs(dy) < 0.15:
                dy = 0.0
            if dx or dy:
                box = el["box"]
                box[0] = round(float(box[0]) + dx, 3)
                box[1] = round(float(box[1]) + dy, 3)
            el["position_source"] = "preview_calibrated"
            el["position_calibration"] = {
                "iteration": iteration,
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "target_ink": [int(round(v)) for v in target],
                "rendered_ink": [
                    round(rx1, 3), round(ry1, 3),
                    round(rx2, 3), round(ry2, 3),
                ],
            }
            report.append({
                "slide": s_idx + 1,
                "iteration": iteration,
                "name": el.get("name"),
                "text": el.get("text"),
                "status": "calibrated",
                "dx": round(dx, 3),
                "dy": round(dy, 3),
            })
    return report


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _build_pptx(layout_path: Path, out_path: Path,
                assets_root: Path | None) -> None:
    _builder.run(
        layout=str(layout_path),
        out=str(out_path),
        assets_root=str(assets_root) if assets_root else None,
    )


def _render_pptx(pptx_path: Path, out_dir: Path, dpi: int) -> None:
    _render.run(
        pptx=str(pptx_path), out_dir=str(out_dir),
        dpi=dpi, verbose=False,
    )


def main() -> int:
    args = parse_args()
    layout_path = Path(args.layout)
    source_dir = Path(args.source_dir)
    work_dir = Path(args.work_dir)
    assets_root = Path(args.assets_root) if args.assets_root else work_dir
    out_layout = Path(args.out_layout) if args.out_layout else layout_path
    debug_dir = work_dir / "debug" / "text_position_calibration"
    debug_dir.mkdir(parents=True, exist_ok=True)

    layout = json.loads(layout_path.read_text(encoding="utf-8-sig"))
    all_reports: list[dict[str, Any]] = []

    for iteration in range(1, max(1, int(args.iterations)) + 1):
        iter_dir = debug_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        cal_layout, colours = _calibration_layout(layout)
        cal_layout_path = iter_dir / "calibration.layout.json"
        cal_pptx_path = iter_dir / "calibration.pptx"
        cal_layout_path.write_text(
            json.dumps(cal_layout, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _build_pptx(cal_layout_path, cal_pptx_path, assets_root)
        _render_pptx(cal_pptx_path, iter_dir, args.dpi)
        report = _apply_iteration(
            layout, colours, iter_dir / "previews", source_dir,
            float(args.max_shift), iteration,
        )
        all_reports.extend(report)

    out_layout.write_text(json.dumps(layout, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    report_path = debug_dir / "position_calibration_report.json"
    report_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    moved = sum(1 for r in all_reports
                if r.get("status") == "calibrated"
                and (r.get("dx") or r.get("dy")))
    print(json.dumps({
        "layout": str(out_layout),
        "report": str(report_path),
        "records": len(all_reports),
        "moved": moved,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
