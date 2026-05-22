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

import inventory_to_layout as itl  # noqa: E402


class InventoryTextBBoxTests(unittest.TestCase):
    def test_non_numeric_leading_glyph_is_not_trimmed_as_icon(self) -> None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:] = np.array([255, 255, 255], dtype=np.uint8)
        red = (55, 55, 205)
        bbox = (100, 100, 190, 134)
        cv2.rectangle(img, (106, 104), (128, 130), red, -1)
        cv2.rectangle(img, (146, 106), (156, 130), red, -1)
        cv2.rectangle(img, (162, 106), (176, 130), red, 3)

        trimmed = itl._trim_short_stat_text_bbox("前10名", bbox, img)

        self.assertEqual(trimmed, bbox)

    def test_mixed_label_line_keeps_suffix_black(self) -> None:
        img = np.full((90, 260, 3), 250, dtype=np.uint8)
        text = "标签：正文内容"
        boxes = []
        x = 12
        for idx, ch in enumerate(text):
            w = 8 if ch == "：" else 22
            boxes.append([x, 18, x + w, 55])
            color = (152, 71, 5) if idx <= 2 else (34, 34, 34)
            y1, y2 = (22, 50) if idx <= 2 else (28, 44)
            cv2.rectangle(img, (x + 3, y1), (x + w - 3, y2), color, -1)
            x += w + 3

        style = itl.detect_text_style(
            [12, 18, x - 3, 55],
            img,
            text=text,
            char_boxes=boxes,
        )

        self.assertEqual(
            style.get("runs"),
            [
                {"text": "标签：", "color": "#054798"},
                {"text": "正文内容", "color": "#222222"},
            ],
        )

    def test_colon_label_mixed_size_preserves_suffix_regular(self) -> None:
        img = np.full((90, 260, 3), 250, dtype=np.uint8)
        text = "标签：正文内容"
        boxes = []
        x = 12
        for idx, ch in enumerate(text):
            w = 8 if ch == "：" else 22
            boxes.append([x, 18, x + w, 55])
            color = (152, 71, 5) if idx <= 2 else (34, 34, 34)
            y1, y2 = (22, 50) if idx <= 2 else (30, 44)
            cv2.rectangle(img, (x + 3, y1), (x + w - 3, y2), color, -1)
            x += w + 3
        runs = [
            {"text": "标签：", "color": "#054798"},
            {"text": "正文内容", "color": "#222222"},
        ]

        original_fit = itl.fit_text_render

        def fake_fit(text, source, bbox, *, initial_size, initial_bold,
                     initial_font, pt_per_px):
            return {
                "font": initial_font,
                "size": int(initial_size),
                "bold": bool(initial_bold),
                "box": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "score": 0.0,
                "target_ink": list(bbox),
                "render_ink": [0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]],
            }

        try:
            itl.fit_text_render = fake_fit
            mixed = itl.mixed_size_runs_from_char_boxes(
                text,
                img,
                (12, 18, x - 3, 55),
                boxes,
                runs,
                "#054798",
                bold=False,
                pt_per_px=0.75,
            )
        finally:
            itl.fit_text_render = original_fit

        self.assertIsNotNone(mixed)
        out_runs = mixed["runs"]
        self.assertGreater(out_runs[0]["size"], out_runs[1]["size"])
        self.assertTrue(out_runs[0]["bold"])
        self.assertFalse(out_runs[1]["bold"])
        self.assertEqual(out_runs[1]["color"], "#222222")


if __name__ == "__main__":
    unittest.main()
