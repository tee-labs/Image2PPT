"""Icon and sub-shape extraction inside an already-detected parent component.

This package isolates the two operations that share a single pattern —
*find a child element inside a parent component crop, then patch the
parent's cleaned image so the parent's downstream asset crop comes out
without the child baked in*:

- :mod:`icon.detect` — three detectors (filled internal shapes, white
  pictograms on dark cards, sparse coloured line-art) that locate child
  bboxes and emit pixel-fill jobs for the parent crop.
- :mod:`icon.inpaint` — :func:`inpaint_region_inplace`, the helper that
  applies those fill jobs with flat modal fill on locally uniform
  backgrounds and ``cv2.inpaint`` on non-uniform ones, so extracted icons
  do not leave ghosted colour fringes in the parent crop.

The detectors expect the caller to pre-filter OCR items down to actual
text glyphs (drop anything for which ``should_preserve_visual`` returns
True) and pass that as ``ocr_text_items``.
"""
from .detect import (
    detect_internal_shapes,
    detect_line_art_subicons,
    detect_white_subicons,
)
from .inpaint import inpaint_region_inplace

__all__ = [
    "detect_internal_shapes",
    "detect_line_art_subicons",
    "detect_white_subicons",
    "inpaint_region_inplace",
]
