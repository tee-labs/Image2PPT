#!/usr/bin/env python
"""Render the OCR-review packet as a single annotated PNG per page.

Replaces the old per-crop tile + contact-sheet workflow. The agent
opens ONE image per page (`inventory/page_NN.ocr_review.annotated.png`)
which contains:

  - the original slide on the left, with every review entry drawn as a
    colored bounding box: GREEN (high-consensus), YELLOW (weak/single-
    engine consensus), RED (no consensus)
  - a side legend listing each entry's idx, tier, all 3 engine
    candidates, and the pre-filled `suggested_text`

The agent reads the image, then either:
  - accepts the suggested texts (no edits needed — deck rebuilds with
    the consensus picks), or
  - overrides specific entries by editing `corrected_text` in
    `page_NN.ocr_review.json`

Used as a library module by prepare_ocr.py. The functions here have no
side effects beyond writing the output PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


TIER_RGB = {
    "green":  (40, 170, 70),
    "yellow": (220, 165, 30),
    "red":    (215, 50, 50),
}
TIER_GLYPH = {"green": "G", "yellow": "Y", "red": "R"}


def _load_font(size: int):
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _shorten(s: str, n: int) -> str:
    if s is None:
        s = ""
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_annotated_page(image_path: Path, entries: list[dict],
                          out_path: Path, *,
                          legend_width: int = 720) -> bool:
    """Render the full-page annotated review image.

    `entries` is the list of review entries as written by prepare_ocr.py:
        {
          "idx": int,
          "bbox": [x1, y1, x2, y2],
          "tier": "green" | "yellow" | "red",
          "reason": str,            # e.g. "A_strict_consensus"
          "candidates": {
            "paddle":    {"text": "...", "conf": 0.91},
            "easyocr":   {"text": "...", "conf": 0.30},
            "tesseract": {"text": "...", "conf": 0.10},
          },
          "suggested_text": "...",
          "corrected_text": "...",   # usually = suggested_text on first render
        }

    Returns True if a file was written, False if entries was empty (in
    which case no annotation is needed and we skip the page).
    """
    if not entries:
        return False

    src = Image.open(image_path).convert("RGB")
    page_w, page_h = src.size

    # Sort entries top-to-bottom, left-to-right so legend order matches
    # what the eye scans on the page.
    entries = sorted(entries, key=lambda e: (e["bbox"][1], e["bbox"][0]))

    # ---- Left panel: source page with colored bbox overlays ----
    page_layer = src.copy()
    draw = ImageDraw.Draw(page_layer, "RGBA")
    label_font = _load_font(16)
    for display_pos, e in enumerate(entries, 1):
        x1, y1, x2, y2 = e["bbox"]
        color = TIER_RGB[e["tier"]]
        # Box: thick colored outline, semi-transparent fill in matching color
        draw.rectangle([x1, y1, x2, y2],
                       outline=color + (255,), width=3,
                       fill=color + (30,))
        # Index tag: small colored square with the display number,
        # placed at top-left of the box (clipped inside the page).
        tag_w, tag_h = 28, 20
        tx = max(0, min(page_w - tag_w, x1 - 1))
        ty = max(0, y1 - tag_h)
        if ty < 0:
            ty = y1 + 1
        draw.rectangle([tx, ty, tx + tag_w, ty + tag_h],
                       fill=color + (255,))
        draw.text((tx + 4, ty + 1), f"#{display_pos}",
                  fill=(255, 255, 255), font=label_font)

    # ---- Right panel: legend table ----
    # Layout per row: tier-chip | header | 3 candidate lines | divider
    row_pad_y = 8
    line_h = 18
    rows = []
    for display_pos, e in enumerate(entries, 1):
        cand = e.get("candidates", {})
        pp = cand.get("paddle",      {"text": "", "conf": 0.0})
        ez = cand.get("easyocr",     {"text": "", "conf": 0.0})
        ts = cand.get("tesseract",   {"text": "", "conf": 0.0})
        pc = cand.get("paddle_crop", None)  # may be absent
        rows.append({
            "n": display_pos,
            "idx": e["idx"],
            "tier": e["tier"],
            "reason": e.get("reason", ""),
            "suggested": e.get("suggested_text", ""),
            "pp_text": pp.get("text", ""),
            "pp_conf": pp.get("conf", 0.0),
            "ez_text": ez.get("text", ""),
            "ez_conf": ez.get("conf", 0.0),
            "ts_text": ts.get("text", ""),
            "ts_conf": ts.get("conf", 0.0),
            "pc_text": (pc or {}).get("text", "") if pc is not None else None,
            "pc_conf": (pc or {}).get("conf", 0.0) if pc is not None else None,
        })

    # Calculate legend height: header row + per-entry block.
    # Per-entry block contains: header line + up to 4 candidate lines
    # (paddle / easyocr / tesseract / optional paddle_crop) + suggested.
    has_paddle_crop = any(
        "paddle_crop" in (e.get("candidates") or {}) for e in entries
    )
    title_h = 36
    cand_lines = 4 if has_paddle_crop else 3
    block_h = (1 + cand_lines + 1) * line_h + row_pad_y * 2
    legend_h = title_h + len(rows) * block_h
    canvas_h = max(page_h, legend_h)
    canvas = Image.new("RGB", (page_w + legend_width, canvas_h),
                       (252, 252, 252))
    canvas.paste(page_layer, (0, 0))

    ldraw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    body_font = _load_font(14)
    mono_font = _load_font(14)
    small_font = _load_font(12)

    lx = page_w + 16  # left edge of legend content
    ly = 12
    ldraw.text((lx, ly),
               f"{len(rows)} entries — tiers: "
               f"{sum(1 for r in rows if r['tier']=='green')} 🟢  "
               f"{sum(1 for r in rows if r['tier']=='yellow')} 🟡  "
               f"{sum(1 for r in rows if r['tier']=='red')} 🔴",
               fill=(40, 40, 40), font=title_font)
    ly += title_h

    for r in rows:
        color = TIER_RGB[r["tier"]]
        tier_glyph = TIER_GLYPH[r["tier"]]
        # Block background tint (very faint)
        ldraw.rectangle(
            [lx - 8, ly - 2, page_w + legend_width - 8, ly + block_h - row_pad_y],
            fill=(color[0], color[1], color[2], 16),
            outline=color + (60,),
        )
        # Header: # tier-chip   idx=N  reason
        ldraw.rectangle([lx, ly, lx + 26, ly + 18], fill=color)
        ldraw.text((lx + 6, ly + 1), f"#{r['n']}",
                   fill=(255, 255, 255), font=body_font)
        ldraw.text((lx + 36, ly + 1),
                   f"{tier_glyph}  idx={r['idx']}  {r['reason']}",
                   fill=(60, 60, 60), font=body_font)
        ly += line_h + 2

        # Candidate lines (truncate texts to fit the column width).
        # PP = full-page Paddle, PC = paddle-on-crop (4th engine, only
        # present when the entry was originally RED and rescue ran),
        # EZ = EasyOCR, TS = Tesseract.
        cand_lines = [
            ("PP", r["pp_text"], r["pp_conf"]),
            ("EZ", r["ez_text"], r["ez_conf"]),
            ("TS", r["ts_text"], r["ts_conf"]),
        ]
        if r["pc_text"] is not None:
            cand_lines.insert(1, ("PC", r["pc_text"], r["pc_conf"]))
        for label, txt, conf in cand_lines:
            shown = _shorten(txt, 60)
            ldraw.text((lx + 8, ly),
                       f"{label}: {shown}",
                       fill=(40, 40, 40), font=mono_font)
            ldraw.text((lx + legend_width - 120, ly),
                       f"conf={conf:.2f}",
                       fill=(110, 110, 110), font=small_font)
            ly += line_h

        # Suggested text (pre-filled corrected_text)
        suggested = _shorten(r["suggested"], 70)
        ldraw.text((lx + 8, ly),
                   f"→ suggested: {suggested}",
                   fill=color, font=body_font)
        ly += line_h + row_pad_y

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


__all__ = ["render_annotated_page", "TIER_RGB"]
