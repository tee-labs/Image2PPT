"""In-place inpainting for child-region removal from a parent component crop.

When an icon (or sub-shape) is extracted from its parent's cleaned-image
bbox, the parent's downstream asset crop must come out without the
extracted shape baked in. Solid-colour fill produces a visible square of
wrong colour whenever the parent has even a slightly non-uniform card
fill; cv2.inpaint propagates surrounding pixels into the masked region
and reconstructs the local background continuously.
"""
from __future__ import annotations

import cv2
import numpy as np


def inpaint_region_inplace(local: np.ndarray, mask_in_crop: np.ndarray,
                           radius: int = 6, scale: float = 1.0) -> None:
    """Replace ``mask_in_crop`` pixels in ``local`` with cv2.inpaint output.

    The mask is dilated by ~4 px (at 720-tuned scale) before inpainting so
    anti-aliased fringe just outside the mask edges is overwritten too —
    a smaller dilation leaves a faint halo of the original child behind.
    Both the dilation iteration count and the inpaint radius scale with
    ``scale = h_img / 720``: at 1080p the equivalent halo is ~6 px and
    the radius widens to ~9, so a 1.5× larger image gets a 1.5× larger
    correction footprint and the parent's downstream crop still comes
    out without a visible ghost.

    INPAINT_NS handles larger flat regions noticeably better than TELEA
    when the surrounding card fill is uniform. ``local`` is a view into
    the parent's cleaned-image bbox; we write back via slice assignment
    so the underlying array is modified in place.
    """
    if mask_in_crop.dtype != np.bool_:
        mask_in_crop = mask_in_crop.astype(bool)
    if not mask_in_crop.any():
        return
    inpaint_mask = mask_in_crop.astype(np.uint8) * 255
    iterations = max(1, int(round(4 * scale)))
    inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), np.uint8),
                              iterations=iterations)
    scaled_radius = max(1, int(round(radius * scale)))
    src = np.ascontiguousarray(local)
    inpainted = cv2.inpaint(src, inpaint_mask, scaled_radius, cv2.INPAINT_NS)
    local[:] = inpainted
