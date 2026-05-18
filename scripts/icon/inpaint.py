"""Background-aware patching for child-region removal from parent crops.

When an icon (or sub-shape) is extracted from its parent's cleaned-image
bbox, the parent's downstream asset crop must come out without the
extracted shape baked in. Solid-colour fill produces a visible square of
wrong colour whenever the parent has even a slightly non-uniform card
fill; cv2.inpaint can smear icon edge colours back into a flat card,
leaving a faint ghost. This helper chooses the safer path per region:
flat/modal background fill for locally uniform backgrounds, inpaint for
non-uniform ones.
"""
from __future__ import annotations

import cv2
import numpy as np


def _normalise_fill_color(
    fill_color: np.ndarray | tuple | list | None,
) -> np.ndarray | None:
    if fill_color is None:
        return None
    arr = np.asarray(fill_color, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _modal_bg_from_pixels(
    pixels: np.ndarray,
    fill_color: np.ndarray | None = None,
) -> tuple[np.ndarray | None, bool]:
    """Return (background BGR, is_simple_enough_for_flat_fill)."""
    if pixels.size == 0:
        if fill_color is None:
            return None, False
        return fill_color.astype(np.uint8), True

    pix = pixels.reshape(-1, 3).astype(np.float32)
    explicit_close = None
    if fill_color is not None:
        diff_exp = np.max(np.abs(pix - fill_color[None, :]), axis=1)
        explicit_close = pix[diff_exp <= 34]

    quant = (pix.astype(np.uint8) // 16) * 16
    values, counts = np.unique(
        quant.reshape(-1, 3), axis=0, return_counts=True)
    winner = values[int(np.argmax(counts))].astype(np.float32) + 8.0
    diff = np.max(np.abs(pix - winner[None, :]), axis=1)
    close = pix[diff <= 28]
    if len(close) < max(8, int(0.08 * len(pix))):
        close = pix

    # If the detector supplied the parent fill colour and the local ring
    # contains enough pixels near it, prefer that cluster. It is often
    # cleaner than a quantized mode when the icon fringe leaked into the
    # sampling ring.
    if (explicit_close is not None
            and len(explicit_close) >= max(8, int(0.12 * len(pix)))):
        close = explicit_close

    med = np.median(close, axis=0)
    p10 = np.percentile(close, 10, axis=0)
    p90 = np.percentile(close, 90, axis=0)
    spread = float(np.max(p90 - p10))
    ratio = len(close) / max(1, len(pix))
    simple = ratio >= 0.32 and spread <= 22.0

    if fill_color is not None:
        agrees = float(np.max(np.abs(med - fill_color))) <= 36.0
        simple = simple and (agrees or ratio >= 0.55)

    return med.astype(np.uint8), simple


def _sample_local_background(
    local: np.ndarray,
    patch_mask: np.ndarray,
    fill_color: np.ndarray | tuple | list | None,
    scale: float,
) -> tuple[np.ndarray | None, bool]:
    """Sample the modal colour in a narrow ring around the patch mask."""
    h, w = patch_mask.shape[:2]
    explicit = _normalise_fill_color(fill_color)
    if h == 0 or w == 0:
        if explicit is not None:
            return explicit.astype(np.uint8), True
        return None, False

    mask_u8 = patch_mask.astype(np.uint8) * 255
    ring_iters = max(2, int(round(5 * scale)))
    ring_outer = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8),
                            iterations=ring_iters) > 0
    ring = ring_outer & ~patch_mask
    if int(ring.sum()) < max(12, int(0.002 * h * w)):
        ring = ~patch_mask
    if int(ring.sum()) == 0:
        if explicit is not None:
            return explicit.astype(np.uint8), True
        return None, False
    return _modal_bg_from_pixels(local[ring], explicit)


def inpaint_region_inplace(
    local: np.ndarray,
    mask_in_crop: np.ndarray,
    radius: int = 6,
    scale: float = 1.0,
    fill_color: np.ndarray | tuple | list | None = None,
) -> None:
    """Replace ``mask_in_crop`` pixels in ``local`` with clean background.

    The mask is dilated by ~5 px (at 720-tuned scale) before inpainting so
    anti-aliased fringe just outside the mask edges is overwritten too —
    a smaller dilation leaves a faint halo of the original child behind.
    Both the dilation iteration count and the inpaint radius scale with
    ``scale = h_img / 720``: at 1080p the equivalent halo is ~8 px and
    the radius widens to ~9, so a 1.5× larger image gets a 1.5× larger
    correction footprint and the parent's downstream crop still comes
    out without a visible ghost.

    For flat backgrounds, the dilated footprint is filled with the modal
    local background colour (or a detector-provided ``fill_color`` when
    it agrees with the local ring). This avoids the faint colour smears
    cv2.inpaint can leave behind. For non-uniform backgrounds, INPAINT_NS
    reconstructs the local texture/gradient. ``local`` is usually a view
    into the parent's cleaned-image bbox; slice assignment writes back to
    the underlying array.
    """
    if mask_in_crop.dtype != np.bool_:
        mask_in_crop = mask_in_crop.astype(bool)
    if not mask_in_crop.any():
        return
    inpaint_mask = mask_in_crop.astype(np.uint8) * 255
    iterations = max(1, int(round(5 * scale)))
    inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), np.uint8),
                              iterations=iterations)
    patch_mask = inpaint_mask > 0
    bg, simple_bg = _sample_local_background(
        local, patch_mask, fill_color, scale)
    if simple_bg and bg is not None:
        local[patch_mask] = bg
        return
    scaled_radius = max(1, int(round(radius * scale)))
    src = np.ascontiguousarray(local)
    inpainted = cv2.inpaint(src, inpaint_mask, scaled_radius, cv2.INPAINT_NS)
    local[:] = inpainted
