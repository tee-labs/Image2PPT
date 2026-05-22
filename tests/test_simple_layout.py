"""Simple-mode layout generator: full-page background + OCR text overlays.

These tests pin the JSON schema simple_layout.build_layout emits so
that combine_layouts → build_pptx_from_layout keeps consuming it as
they would a full-pipeline layout. The point isn't to verify font
sizing is pixel-perfect — calibration would refine that — but that the
shape of the dict downstream code reads matches what it expects.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "scripts" / "page"
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

import simple_layout as sl  # noqa: E402


def _ocr_fixture() -> list[dict]:
    return [
        {
            "text": "DeckWeaver",
            "x1": 100, "y1": 50, "x2": 400, "y2": 110,
            "confidence": 0.99,
        },
        {
            "text": "图像转 PPT",
            "x1": 120, "y1": 130, "x2": 380, "y2": 175,
            "confidence": 0.97,
        },
        # Blank-text item should be dropped — erase_text would have
        # left no overlay anyway and a zero-content text box renders
        # as a stray empty frame.
        {"text": "  ", "x1": 0, "y1": 200, "x2": 50, "y2": 220},
        # Degenerate bbox should also be dropped.
        {"text": "x", "x1": 10, "y1": 10, "x2": 10, "y2": 12},
    ]


class SimpleLayoutSchemaTests(unittest.TestCase):
    def test_top_level_keys_match_combine_layouts_expectations(self) -> None:
        layout = sl.build_layout(
            source_width=1280, source_height=720,
            background_image_path="inventory/page_01.clean.png",
            ocr=_ocr_fixture(),
        )
        # Keys that combine_layouts.slide_specs reads. Missing any of
        # these would cause combined.layout.json to inherit defaults.
        for key in ("slide_size", "source_width", "source_height",
                    "background", "elements"):
            self.assertIn(key, layout, f"missing top-level key: {key}")

        self.assertEqual(layout["source_width"], 1280)
        self.assertEqual(layout["source_height"], 720)
        self.assertEqual(layout["slide_size"]["height_in"], 7.5)
        # Width scales with source aspect ratio at 7.5" height.
        self.assertAlmostEqual(
            layout["slide_size"]["width_in"],
            7.5 * (1280 / 720),
            places=4,
        )

    def test_first_element_is_full_page_background_image(self) -> None:
        layout = sl.build_layout(
            source_width=1280, source_height=720,
            background_image_path="inventory/page_01.clean.png",
            ocr=_ocr_fixture(),
        )
        bg = layout["elements"][0]
        self.assertEqual(bg["type"], "image")
        self.assertEqual(bg["path"], "inventory/page_01.clean.png")
        self.assertEqual(bg["box"], [0, 0, 1280, 720])
        # build_pptx_from_layout renders elements in array order. Image
        # first → behind editable text. Document the contract here so a
        # future refactor that reorders elements fails this test.
        for el in layout["elements"][1:]:
            self.assertEqual(el["type"], "text")

    def test_text_elements_carry_fields_pptx_builder_reads(self) -> None:
        layout = sl.build_layout(
            source_width=1280, source_height=720,
            background_image_path="inventory/page_01.clean.png",
            ocr=_ocr_fixture(),
        )
        texts = [e for e in layout["elements"] if e["type"] == "text"]
        # Two valid items; blank and degenerate were dropped.
        self.assertEqual(len(texts), 2)

        for t in texts:
            # build_pptx_from_layout.add_text reads these. If any are
            # missing it falls back to a default — which is fine for
            # font/align/color, but `box` and `text` are load-bearing.
            self.assertIn("text", t)
            self.assertIn("box", t)
            self.assertEqual(len(t["box"]), 4)
            self.assertGreater(t["size"], 0)

        first = texts[0]
        self.assertEqual(first["text"], "DeckWeaver")
        self.assertEqual(first["box"], [100, 50, 300, 60])
        self.assertEqual(first["source_bbox"], [100, 50, 400, 110])

    def test_font_size_scales_with_bbox_and_slide_height(self) -> None:
        # Two bboxes with the same height should map to the same point
        # size regardless of x-position. And doubling slide height
        # should double font size for a fixed bbox.
        ocr = [{"text": "hello", "x1": 0, "y1": 0, "x2": 100, "y2": 40}]
        a = sl.build_layout(
            source_width=1000, source_height=500,
            background_image_path="bg.png", ocr=ocr,
            slide_height_in=7.5,
        )
        b = sl.build_layout(
            source_width=1000, source_height=500,
            background_image_path="bg.png", ocr=ocr,
            slide_height_in=15.0,
        )
        pt_a = a["elements"][1]["size"]
        pt_b = b["elements"][1]["size"]
        self.assertAlmostEqual(pt_b / pt_a, 2.0, places=0)

    def test_write_layout_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ocr_path = tmp_path / "page_01.ocr.json"
            ocr_path.write_text(
                json.dumps(_ocr_fixture(), ensure_ascii=False),
                encoding="utf-8",
            )
            out = tmp_path / "layouts" / "page_01.layout.json"
            stats = sl.write_layout(
                page_num="01",
                source_width=1280, source_height=720,
                clean_rel_path="inventory/page_01.clean.png",
                ocr_path=ocr_path,
                out_layout_path=out,
            )
            self.assertTrue(out.exists())
            self.assertEqual(stats["text"], 2)
            self.assertEqual(stats["image"], 1)
            on_disk = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["source_width"], 1280)
            self.assertEqual(on_disk["elements"][0]["role"], "background")


if __name__ == "__main__":
    unittest.main()
