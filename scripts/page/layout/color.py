"""Colour classification + comparison helpers for inventory_to_layout.

These five helpers normalise anti-aliased per-character samples back to
the canonical PPT palette so two visually-identical chars do not split
into separate runs because their sampled hex strings differ by ±5.
"""
from __future__ import annotations

import numpy as np


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _color_close(c1: str, c2: str, tol: int = 25) -> bool:
    """Per-channel-tolerant comparison of two #RRGGBB strings.

    Anti-aliasing causes slightly different sampled colors for two
    visually-identical text blocks; exact equality would prevent merging
    them.
    """
    if c1 == c2:
        return True
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return abs(r1 - r2) <= tol and abs(g1 - g2) <= tol and abs(b1 - b2) <= tol


def _classify_text_color(r: int, g: int, b: int, bg: np.ndarray) -> str:
    """Map a sampled text-stroke (r,g,b) → normalized hex color.

    The classification matters: anti-aliased samples within ±10 of pure
    black often land at e.g. (18, 32, 28) — without normalization to
    `#222222`, two visually-identical black chars produce different
    hex strings and break run-grouping.
    """
    bg_int = bg.astype(int)
    bg_lum = int(0.114 * bg_int[0] + 0.587 * bg_int[1] + 0.299 * bg_int[2])
    bg_is_dark = bg_lum < 110
    text_lum = int(0.114 * b + 0.587 * g + 0.299 * r)
    if bg_is_dark and text_lum > 170:
        return "#FFFFFF"
    if r > g + 30 and r > b + 30 and r > 150:
        return "#D43E3E"
    # Canonical deck blue. Require a real blue-vs-gray separation; low
    # saturation grey-blue body text such as #7A8191 should remain grey
    # instead of being snapped to the saturated title blue.
    if b > r + 28 and b > g + 18 and b > 80:
        return "#054798"
    if r < 80 and g < 80 and b < 80:
        return "#222222"
    # Two-tier near-black: very-low-saturation samples up to max_ch=130
    # still read as black-ish on light backgrounds — per-char sampling on
    # thin strokes routinely lands at (108,108,110) or (122,125,125) on
    # what's visually solid dark text. Without absorbing these into
    # #222222, run-grouping splits a single-color sentence into
    # alternating chunks of #6C6C6E / #222222 / #727375 / ...
    max_ch = max(r, g, b)
    min_ch = min(r, g, b)
    if max_ch < 130 and (max_ch - min_ch) < 12:
        return "#222222"
    return f"#{r:02X}{g:02X}{b:02X}"


def _sampled_color_hex(pixels: np.ndarray, bg: np.ndarray) -> str:
    """Return a canonical RGB hex colour from sampled BGR stroke pixels."""
    m = np.median(pixels, axis=0).astype(int)
    b, g, r = int(m[0]), int(m[1]), int(m[2])
    return _classify_text_color(r, g, b, bg)


def _normalize_run_colors(runs: list[dict], fallback_color: str,
                          bg: np.ndarray) -> None:
    """Snap anti-aliased run samples back to the line-level colour."""
    if not runs:
        return
    for run in runs:
        raw = run.get("color")
        if not isinstance(raw, str) or not raw.startswith("#"):
            run["color"] = fallback_color
            continue
        try:
            rr, gg, bb = _hex_to_rgb(raw)
        except (TypeError, ValueError):
            run["color"] = fallback_color
            continue
        classified = _classify_text_color(rr, gg, bb, bg)
        if classified == fallback_color or _color_close(
                raw, fallback_color, tol=70):
            run["color"] = fallback_color
