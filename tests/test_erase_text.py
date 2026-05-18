from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PAGE_DIR = SCRIPTS_DIR / "page"
for path in (SCRIPTS_DIR, PAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from erase_text import erase_text  # noqa: E402


BLUE = np.array([152, 71, 5], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)


def _slide_with_text_and_dash() -> np.ndarray:
    img = np.zeros((70, 220, 3), dtype=np.uint8)
    img[:] = WHITE
    cv2.putText(img, "TITLEX", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, tuple(int(v) for v in BLUE), 2, cv2.LINE_AA)
    for x in range(8, 188, 16):
        cv2.line(img, (x, 43), (x + 8, 43),
                 tuple(int(v) for v in BLUE), 2, cv2.LINE_AA)
    return img


def _blue_count(region: np.ndarray) -> int:
    return int((np.abs(region.astype(np.int16) - WHITE.astype(np.int16))
                .max(axis=2) > 30).sum())


def _line_preserved(before: np.ndarray, after: np.ndarray) -> float:
    before_line = before[40:47, 5:195]
    after_line = after[40:47, 5:195]
    line_mask = (
        np.abs(before_line.astype(np.int16) - WHITE.astype(np.int16))
        .max(axis=2) > 30
    )
    delta = np.abs(after_line.astype(np.int16) - before_line.astype(np.int16))
    same = (delta.max(axis=2) <= 3) & line_mask
    return float(same.sum()) / max(1, int(line_mask.sum()))


class EraseTextBoundaryTests(unittest.TestCase):
    def test_word_boxes_prevent_same_colour_dash_from_being_erased(self) -> None:
        img = _slide_with_text_and_dash()
        item = {
            "text": "TITLEX",
            "x1": 8, "y1": 7, "x2": 118, "y2": 47,
            "confidence": 0.99,
            "words": ["TITLEX"],
            "word_boxes": [[10, 8, 116, 32]],
        }

        cleaned, _mask, _decisions = erase_text(img, [item])

        self.assertLess(_blue_count(cleaned[6:34, 8:120]),
                        _blue_count(img[6:34, 8:120]) * 0.25)
        self.assertGreaterEqual(_line_preserved(img, cleaned), 0.96)

    def test_thin_row_band_filter_preserves_dash_without_word_boxes(self) -> None:
        img = _slide_with_text_and_dash()
        item = {
            "text": "TITLEX",
            "x1": 8, "y1": 7, "x2": 118, "y2": 47,
            "confidence": 0.99,
        }

        cleaned, _mask, _decisions = erase_text(img, [item])

        self.assertLess(_blue_count(cleaned[6:34, 8:120]),
                        _blue_count(img[6:34, 8:120]) * 0.25)
        self.assertGreaterEqual(_line_preserved(img, cleaned), 0.94)

    def test_word_box_guard_keeps_erasing_thin_glyph_bands(self) -> None:
        img = np.zeros((70, 180, 3), dtype=np.uint8)
        img[:] = WHITE
        cv2.putText(img, "TITLEX", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, tuple(int(v) for v in BLUE), 2, cv2.LINE_AA)
        cv2.line(img, (16, 38), (112, 38),
                 tuple(int(v) for v in BLUE), 2, cv2.LINE_AA)
        item = {
            "text": "TITLEX",
            "x1": 8, "y1": 7, "x2": 118, "y2": 43,
            "confidence": 0.99,
            "words": ["TITLEX"],
            "word_boxes": [[10, 8, 116, 40]],
            "_force_text": True,
        }

        cleaned, _mask, _decisions = erase_text(img, [item])

        self.assertLess(_blue_count(cleaned[6:43, 8:120]),
                        _blue_count(img[6:43, 8:120]) * 0.08)

    def test_word_box_guard_erases_horizontally_tight_bold_text(self) -> None:
        img = np.zeros((60, 180, 3), dtype=np.uint8)
        img[:] = WHITE
        red = np.array([55, 55, 205], dtype=np.uint8)
        cv2.putText(img, "NEV", (24, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, tuple(int(v) for v in red), 3, cv2.LINE_AA)
        item = {
            "text": "NEV",
            "x1": 20, "y1": 10, "x2": 96, "y2": 43,
            "confidence": 0.99,
            "words": ["NEV"],
            # Deliberately too tight on x, matching Paddle's occasional
            # bold-title behaviour.
            "word_boxes": [[32, 12, 86, 42]],
            "_force_text": True,
        }

        cleaned, _mask, _decisions = erase_text(img, [item])

        before = np.abs(img[8:46, 18:100].astype(np.int16)
                        - WHITE.astype(np.int16)).max(axis=2) > 30
        after = np.abs(cleaned[8:46, 18:100].astype(np.int16)
                       - WHITE.astype(np.int16)).max(axis=2) > 30
        self.assertLess(int(after.sum()), int(before.sum()) * 0.08)


if __name__ == "__main__":
    unittest.main()
