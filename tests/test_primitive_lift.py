"""Primitive shapes lifted out of merged components become native shapes.

Regression coverage for the issue "原生形状没有做好": a circle / rectangle
that touches other content merges into one big connected component and
used to be flattened into the background PNG forever. These tests lock
in the detection-side lifts:

* detect_internal_shapes — rings, box frames and big decorative circles
  inside composite parents surface as internal-shape candidates, while
  contaminated silhouettes (ring fused with a line) stay rejected.
* InventoryBuilder — whole-slide background components are scanned for
  internal primitives instead of being skipped.
* classify_filled_shape — hollow rectangular frames classify to rect /
  round_rect with transparent fill (the ring path only handled
  near-square annuli).
* LayoutBuilder — images contained inside a front native shape render
  after it, so a lifted shape never hides the icon lifted from inside
  it.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "scripts" / "page"
SCRIPTS_DIR = ROOT / "scripts"
for _p in (str(PAGE_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from icon import detect_internal_shapes  # noqa: E402
from layout.outline import classify_filled_shape  # noqa: E402

ORANGE_BGR = (60, 110, 200)     # → #C86E3C
NAVY_BGR = (120, 80, 40)        # → #285078
GRAY_BGR = (90, 90, 90)         # → #5A5A5A


def _white(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _rounded_frame(w: int, h: int, r: int, t: int,
                   color=GRAY_BGR) -> np.ndarray:
    """Hollow rounded-rect frame whose stroke touches the crop edge."""
    img = _white(w, h)
    outer = np.zeros((h, w), np.uint8)
    cv2.rectangle(outer, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(outer, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r),
                   (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(outer, (cx, cy), r, 255, -1)
    inner = np.zeros((h, w), np.uint8)
    r2 = max(0, r - t)
    cv2.rectangle(inner, (t + r2, t), (w - 1 - t - r2, h - 1 - t), 255, -1)
    cv2.rectangle(inner, (t, t + r2), (w - 1 - t, h - 1 - t - r2), 255, -1)
    for cx, cy in ((t + r2, t + r2), (w - 1 - t - r2, t + r2),
                   (t + r2, h - 1 - t - r2),
                   (w - 1 - t - r2, h - 1 - t - r2)):
        cv2.circle(inner, (cx, cy), r2, 255, -1)
    img[outer > 0] = color
    img[inner > 0] = 255
    return img


class DetectInternalPrimitiveTests(unittest.TestCase):
    def test_ring_in_composite_parent_found(self) -> None:
        """Issue reproduction: circle outline + card + solid circle all
        living inside one merged parent component."""
        img = _white(640, 400)
        cv2.circle(img, (150, 150), 70, ORANGE_BGR, 5)
        cv2.rectangle(img, (60, 280), (259, 379), (250, 230, 200), -1)
        cv2.circle(img, (480, 120), 60, NAVY_BGR, -1)
        cv2.circle(img, (480, 120), 22, (255, 255, 255), -1)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 640, 400,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        boxes = [tuple(s) for s in shapes]
        self.assertIn((420, 60, 541, 181), boxes)      # navy circle
        self.assertIn((60, 280, 260, 380), boxes)      # pale card

    def test_standalone_ring_found(self) -> None:
        img = _white(300, 300)
        cv2.circle(img, (150, 150), 80, ORANGE_BGR, 4)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 300, 300,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        self.assertEqual(len(shapes), 1)
        self.assertAlmostEqual(shapes[0][2] - shapes[0][0], 165, delta=6)

    def test_big_decorative_ring_found(self) -> None:
        """Rings far larger than the 220 px chip window must still lift."""
        img = _white(600, 600)
        cv2.circle(img, (300, 300), 250, ORANGE_BGR, 6)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 600, 600,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        self.assertEqual(len(shapes), 1)
        self.assertGreater(shapes[0][2] - shapes[0][0], 480)

    def test_hollow_rect_frame_found(self) -> None:
        img = _white(640, 400)
        cv2.rectangle(img, (50, 50), (300, 200), GRAY_BGR, 3)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 640, 400,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        self.assertTrue(any(
            abs((s[2] - s[0]) - 254) < 6 and abs((s[3] - s[1]) - 154) < 6
            for s in shapes))

    def test_ring_fused_with_line_rejected(self) -> None:
        """A ring merged with a touching line is not a clean primitive —
        leave it on the parent (no bogus native oval)."""
        img = _white(640, 400)
        cv2.circle(img, (150, 150), 70, ORANGE_BGR, 5)
        cv2.line(img, (218, 150), (400, 150), (150, 150, 150), 3)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 640, 400,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        self.assertEqual(shapes, [])

    def test_uniform_card_parent_badges_still_found(self) -> None:
        """Old working path: uniform pale card parent with chips."""
        img = np.full((300, 500, 3), (250, 235, 215), np.uint8)
        cv2.circle(img, (80, 150), 30, (200, 120, 40), -1)
        cv2.rectangle(img, (150, 120), (230, 180), (80, 80, 200), -1)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 500, 300,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        self.assertEqual(len(shapes), 2)

    def test_lone_big_filled_circle_on_slide_bg_found(self) -> None:
        """Modal-seed inversion: a 320 px filled circle on a white slide
        makes the circle's own fill the modal foreground colour; seeding
        with it erased the shape. The slide-bg retry must recover it —
        dense primitives also need the wider 600 px window."""
        img = _white(1280, 720)
        cv2.circle(img, (950, 360), 160, ORANGE_BGR, -1)
        shapes, _ = detect_internal_shapes(
            img, 0, 0, 1280, 720,
            min_dim=20, max_dim=220, min_area=400, scale=1.0)
        self.assertEqual(len(shapes), 1)
        x1, y1, x2, y2 = shapes[0]
        self.assertGreater(x2 - x1, 300)
        self.assertAlmostEqual(y2 - y1, x2 - x1, delta=8)


class QuadRingClassificationTests(unittest.TestCase):
    def test_hollow_rect_frame_is_transparent_rect(self) -> None:
        img = _rounded_frame(300, 120, 2, 3)
        kind, fill, line, radius, line_px = classify_filled_shape(img)
        self.assertEqual(kind, "rect")
        self.assertIsNone(fill)
        self.assertEqual(line, "#5A5A5A")
        self.assertGreater(line_px, 1.0)

    def test_hollow_rounded_frame_is_transparent_round_rect(self) -> None:
        kind, fill, _line, radius, _line_px = classify_filled_shape(
            _rounded_frame(300, 120, 18, 3))
        self.assertEqual(kind, "round_rect")
        self.assertIsNone(fill)
        self.assertGreaterEqual(radius, 0.08)
        self.assertLessEqual(radius, 0.4)

    def test_square_rounded_frame_is_transparent_round_rect(self) -> None:
        kind, fill, _line, _radius, _line_px = classify_filled_shape(
            _rounded_frame(160, 160, 30, 5, color=(200, 120, 40)))
        self.assertEqual(kind, "round_rect")
        self.assertIsNone(fill)

    def test_l_shape_rejected(self) -> None:
        img = _white(300, 120)
        cv2.rectangle(img, (5, 5), (294, 40), GRAY_BGR, -1)
        cv2.rectangle(img, (5, 5), (40, 114), GRAY_BGR, -1)
        self.assertIsNone(classify_filled_shape(img))

    def test_broken_frame_rejected(self) -> None:
        img = _rounded_frame(300, 120, 2, 3)
        cv2.rectangle(img, (140, 0), (160, 30), (255, 255, 255), -1)
        self.assertIsNone(classify_filled_shape(img))

    def test_frame_with_content_rejected(self) -> None:
        img = _rounded_frame(300, 120, 2, 3)
        cv2.rectangle(img, (100, 40), (200, 80), (200, 120, 40), -1)
        self.assertIsNone(classify_filled_shape(img))


class BackgroundScanTests(unittest.TestCase):
    def _build_inventory(self, td: Path, img: np.ndarray):
        from inventory.builder import InventoryBuilder
        src = td / "page_01.png"
        cv2.imwrite(str(src), img)
        inv = td / "inventory.json"
        args = type("Args", (), {
            "clean": str(src),
            "source": str(src),
            "ocr": str(td / "ocr.json"),
            "out": str(inv),
            "masks_dir": None,
            "debug_dir": None,
            "min_area": 80,
            "dilate": 6,
            "split_gap": 12,
        })()
        (td / "ocr.json").write_text("[]", encoding="utf-8")
        InventoryBuilder(args).build_and_write()
        return json.loads(inv.read_text(encoding="utf-8"))

    def test_background_scanned_for_primitives(self) -> None:
        """A circle ring + rect card merged into a full-bleed background
        panel component must lift as internal shapes, not stay baked in
        the background PNG."""
        img = _white(1280, 720)
        # Full-bleed rounded panel → one whole-slide component. Its pale
        # saturated fill keeps it foreground for the whitespace-split
        # pass too, so the ring/card inside stay merged into it.
        panel = np.zeros((720, 1280), np.uint8)
        cv2.rectangle(panel, (44, 14), (1235, 705), 255, -1)
        cv2.rectangle(panel, (14, 44), (1265, 675), 255, -1)
        for cx, cy in ((44, 44), (1235, 44), (44, 675), (1235, 675)):
            cv2.circle(panel, (cx, cy), 30, 255, -1)
        img[panel > 0] = (208, 238, 248)
        cv2.circle(img, (200, 200), 90, ORANGE_BGR, 6)           # ring
        cv2.rectangle(img, (500, 120), (649, 209),
                      (160, 200, 240), -1)                       # card
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            inventory = self._build_inventory(td, img)
            internal = [el for el in inventory
                        if el.get("role") == "internal"]
            boxes = [tuple(el["bbox"]) for el in internal]
            ring = any(b[2] - b[0] > 150 and abs(
                (b[2] - b[0]) - (b[3] - b[1])) < 12 for b in boxes)
            card = any(abs((b[2] - b[0]) - 149) < 10
                       and abs((b[3] - b[1]) - 89) < 10 for b in boxes)
            self.assertTrue(ring, f"ring not lifted: {boxes}")
            self.assertTrue(card, f"card not lifted: {boxes}")


class FrontShapeZOrderTests(unittest.TestCase):
    def test_contained_icon_renders_after_front_shape(self) -> None:
        """Icon lifted from inside a lifted circle must not be hidden
        by the circle's native shape."""
        from layout.builder import LayoutBuilder
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            src = td / "page_01.png"
            img = _white(540, 960)
            cv2.circle(img, (200, 200), 80, NAVY_BGR, -1)
            cv2.line(img, (185, 200), (215, 200), (255, 255, 255), 6)
            cv2.line(img, (200, 185), (200, 215), (255, 255, 255), 6)
            cv2.imwrite(str(src), img)
            # Real work-dir layout: cleaned.png + cleaned.text_only.png
            # (pre-inpaint copy). Sub-icon crops come from the sidecar.
            cv2.imwrite(str(td / "cleaned.png"), _white(540, 960))
            cv2.imwrite(str(td / "cleaned.text_only.png"), img)
            inventory = [
                {"id": "v000", "type": "image",
                 "bbox": [120, 120, 280, 280],
                 "source": "source", "role": "internal"},
                {"id": "v001", "type": "image",
                 "bbox": [170, 170, 230, 230],
                 "source": "source", "role": "subicon"},
            ]
            inv_path = td / "inventory.json"
            inv_path.write_text(json.dumps(inventory), encoding="utf-8")
            args = type("Args", (), {
                "inventory": str(inv_path),
                "source": str(src),
                "cleaned": str(td / "cleaned.png"),
                "out_assets_dir": str(td / "assets" / "page_01"),
                "asset_prefix": "assets/page_01",
                "out_manifest": str(td / "m.json"),
                "out_layout": str(td / "l.json"),
                "slide_width_in": None,
                "slide_height_in": 7.5,
            })()
            builder = LayoutBuilder(args)
            builder.build()
            builder.write()
            self.assertEqual(len(builder.front_shape_elements), 1)
            self.assertEqual(builder.front_shape_elements[0]["shape"],
                             "oval")
            # v001 (subicon) moved out of image_elements into
            # front_images so it renders after the oval.
            self.assertEqual([el["name"]
                              for el in builder.image_elements], [])
            self.assertEqual([el["name"]
                              for el in builder.front_images], ["v001"])
            layout = json.loads(
                (td / "l.json").read_text(encoding="utf-8"))
            order = [(el["type"], el["name"])
                     for el in layout["elements"]]
            self.assertEqual(order, [("shape", "v000"),
                                     ("image", "v001")])


if __name__ == "__main__":
    unittest.main()
