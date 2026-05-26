"""Shared canvas / bbox background sampling.

Replaces the five overlapping background-sampling implementations:
  - inventory_to_layout.estimate_slide_background_hex
  - build_inventory._estimate_background_bgr
  - inventory_to_layout._sample_background_for_bbox
  - inventory_to_layout._simple_background_strip (kept local; uses extra heuristics)
  - run_pipeline._region_background (kept local; samples a row strip)

The two corner-vote variants are now `estimate_canvas_bgr` and a thin
hex wrapper. The bbox-ring variant is `sample_background_for_bbox`.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def estimate_canvas_bgr(img: np.ndarray) -> np.ndarray:
    """Estimate the slide canvas colour from corner patches, BGR.

    Generated slide screenshots usually have a solid canvas behind the
    content. Sampling corners keeps dark themes dark in the rebuilt PPTX
    and avoids painting letterbox margins white for portrait/square pages.
    """
    h, w = img.shape[:2]
    patch = max(8, min(h, w) // 16)
    samples = [
        img[:patch, :patch],
        img[:patch, w - patch:w],
        img[h - patch:h, :patch],
        img[h - patch:h, w - patch:w],
    ]
    pixels = np.concatenate([s.reshape(-1, 3) for s in samples if s.size])
    if pixels.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    quant = (pixels // 16) * 16
    winner, _ = Counter(map(tuple, quant)).most_common(1)[0]
    winner_arr = np.array(winner, dtype=np.int16)
    diff = np.abs(pixels.astype(np.int16) - winner_arr).max(axis=1)
    close = pixels[diff <= 24]
    return np.median(close if len(close) else pixels, axis=0).astype(np.uint8)


def estimate_canvas_hex(img: np.ndarray) -> str:
    """Same as `estimate_canvas_bgr` but returns `#RRGGBB`."""
    b, g, r = (int(v) for v in estimate_canvas_bgr(img))
    return f"#{r:02X}{g:02X}{b:02X}"


def sample_background_for_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Sample the background colour just outside a text bbox.

    A 6-px ring offset 2 px from the bbox is used so anti-aliased
    character edges do not pollute the sampled background median.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_img, w_img = img.shape[:2]
    x1 = max(0, min(w_img, x1)); x2 = max(0, min(w_img, x2))
    y1 = max(0, min(h_img, y1)); y2 = max(0, min(h_img, y2))
    gap = 2
    ring = 6
    inner_top = max(0, y1 - gap)
    inner_bot = min(h_img, y2 + gap)
    inner_left = max(0, x1 - gap)
    inner_right = min(w_img, x2 + gap)
    samples = []
    if inner_top - ring >= 0:
        samples.append(img[inner_top - ring:inner_top, inner_left:inner_right])
    if inner_bot + ring <= h_img:
        samples.append(img[inner_bot:inner_bot + ring, inner_left:inner_right])
    if inner_left - ring >= 0:
        samples.append(img[inner_top:inner_bot, inner_left - ring:inner_left])
    if inner_right + ring <= w_img:
        samples.append(img[inner_top:inner_bot, inner_right:inner_right + ring])
    if samples:
        pixels = np.concatenate([s.reshape(-1, 3) for s in samples if s.size])
        if len(pixels):
            quant = (pixels // 16) * 16
            mode_q = np.array(Counter(map(tuple, quant)).most_common(1)[0][0])
            diff = np.abs(pixels.astype(int) - mode_q).max(axis=1)
            close = pixels[diff <= 30]
            if len(close) >= max(8, int(len(pixels) * 0.2)):
                return np.median(close, axis=0)
            return np.median(pixels, axis=0)
    if x2 > x1 and y2 > y1:
        region = img[y1:y2, x1:x2]
        if region.size:
            border = np.concatenate([
                region[:1, :].reshape(-1, 3),
                region[-1:, :].reshape(-1, 3),
                region[:, :1].reshape(-1, 3),
                region[:, -1:].reshape(-1, 3),
            ])
            if len(border):
                return np.median(border, axis=0)
    return np.array([255.0, 255.0, 255.0])
