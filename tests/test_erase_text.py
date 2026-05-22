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

from erase_text import erase_text, fill_color  # noqa: E402
from _heuristics import is_likely_icon  # noqa: E402


BLUE = np.array([152, 71, 5], dtype=np.uint8)
LIGHT_BLUE = np.array([253, 230, 208], dtype=np.uint8)
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
    def test_fill_color_keeps_tinted_pill_behind_dense_dark_text(self) -> None:
        img = np.zeros((80, 260, 3), dtype=np.uint8)
        img[:] = WHITE
        cv2.rectangle(img, (30, 25), (230, 60),
                      tuple(int(v) for v in LIGHT_BLUE), -1)
        # One connected, saturated "dense glyph" mass. It covers most of
        # the OCR bbox but does not behave like a container fill because it
        # does not reach enough bbox edges.
        cv2.rectangle(img, (70, 39), (190, 54),
                      tuple(int(v) for v in BLUE), -1)

        bg = fill_color(img, 60, 32, 200, 56)

        self.assertLessEqual(
            int(np.max(np.abs(bg.astype(np.int16)
                              - LIGHT_BLUE.astype(np.int16)))),
            5,
        )

    def test_fill_color_still_detects_tight_colored_container(self) -> None:
        img = np.zeros((80, 220, 3), dtype=np.uint8)
        img[:] = WHITE
        cv2.rectangle(img, (50, 24), (170, 56),
                      tuple(int(v) for v in BLUE), -1)

        bg = fill_color(img, 50, 24, 170, 56)

        self.assertLessEqual(
            int(np.max(np.abs(bg.astype(np.int16)
                              - BLUE.astype(np.int16)))),
            5,
        )

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

    def test_short_text_word_boxes_keep_edge_glyphs_erasable(self) -> None:
        img = np.zeros((80, 170, 3), dtype=np.uint8)
        img[:] = WHITE
        red = np.array([55, 55, 205], dtype=np.uint8)
        cv2.rectangle(img, (18, 25), (40, 53), tuple(int(v) for v in red), -1)
        cv2.rectangle(img, (54, 28), (65, 52), tuple(int(v) for v in red), -1)
        cv2.rectangle(img, (70, 27), (84, 53), tuple(int(v) for v in red), 3)
        cv2.rectangle(img, (98, 25), (120, 53), tuple(int(v) for v in red), -1)
        item = {
            "text": "前10名",
            "x1": 16, "y1": 23, "x2": 122, "y2": 55,
            "confidence": 0.99,
            "words": ["前", "10", "名"],
            "word_boxes": [[18, 25, 40, 53], [54, 25, 84, 53],
                           [98, 25, 120, 53]],
            "_force_text": True,
        }

        cleaned, _mask, _decisions = erase_text(img, [item])

        before = np.abs(img[20:58, 12:126].astype(np.int16)
                        - WHITE.astype(np.int16)).max(axis=2) > 30
        after = np.abs(cleaned[20:58, 12:126].astype(np.int16)
                       - WHITE.astype(np.int16)).max(axis=2) > 30
        self.assertLess(int(after.sum()), int(before.sum()) * 0.08)

    def test_repeated_decorative_symbols_are_preserved_as_icon(self) -> None:
        img = np.zeros((80, 180, 3), dtype=np.uint8)
        img[:] = WHITE
        item = {
            "text": "★★★",
            "x1": 40, "y1": 20, "x2": 100, "y2": 44,
            "confidence": 0.99,
        }

        is_icon, reason = is_likely_icon(item, [item], img)

        self.assertTrue(is_icon)
        self.assertEqual(reason, "decorative_symbol")


if __name__ == "__main__":
    unittest.main()
