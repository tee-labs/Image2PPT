"""Integration motif matrix: shapes/lines must lift to native elements.

The unit suites call the classifiers directly on synthetic crops; every
regression this matrix guards lived in the inventory → role → builder
interplay (merged components, role gates, erase remnants) that direct
classifier tests never exercise. These tests run the REAL per-page
pipeline (erase_text → build_inventory → inventory_to_layout) on
synthetic slides with hand-written OCR JSON — no OCR engine needed —
and assert what each drawn motif became: a native shape / line, or an
image (only where a PNG is the accepted outcome).

Known accepted images: a soft-shadow card (gradient content has no
native equivalent) and the border frame of a header-band card (the
band's erase overreaches the parent border). Everything else must lift.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "scripts" / "page"
SCRIPTS_DIR = ROOT / "scripts"
for _p in (str(PAGE_DIR), str(SCRIPTS_DIR), str(SCRIPTS_DIR / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W, H = 1280, 720

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def _draw_text(img, cx, cy, text, size, fill, ocr_items):
    font = _load_font(size)
    if font is None:
        return
    l, t, r, b = img.textbbox((0, 0), text, font=font)
    x = int(cx - (r - l) / 2 - l)
    y = int(cy - (b - t) / 2 - t)
    img.text((x, y), text, font=font, fill=fill)
    ocr_items.append({"x1": x, "y1": y, "x2": x + (r - l), "y2": y + (b - t),
                      "text": text, "confidence": 0.99})


def _rr(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline,
                        width=width if outline else 0)


def _canvas():
    return Image.new("RGB", (W, H), (255, 255, 255))


def _page_shapes_plain():
    img = _canvas()
    d = ImageDraw.Draw(img)
    motifs = {}
    motifs["filled_round_rect"] = (60, 40, 320, 180)
    _rr(d, (60, 40, 320, 180), 16, fill=(40, 100, 200))
    motifs["filled_rect"] = (360, 40, 620, 180)
    d.rectangle((360, 40, 620, 180), fill=(138, 138, 138))
    motifs["pale_card"] = (660, 40, 920, 180)
    _rr(d, (660, 40, 920, 180), 16, fill=(250, 247, 245))
    motifs["circle"] = (980, 50, 1100, 170)
    d.ellipse((980, 50, 1100, 170), fill=(40, 100, 200))
    motifs["hollow_rect"] = (60, 240, 320, 380)
    d.rectangle((60, 240, 320, 380), outline=(85, 85, 85), width=3)
    motifs["hollow_round_rect"] = (360, 240, 620, 380)
    _rr(d, (360, 240, 620, 380), 16, outline=(85, 85, 85), width=3)
    motifs["dashed_frame"] = (660, 240, 920, 380)
    x = 672
    while x < 900:  # dashed top/bottom
        d.line((x, 242, min(x + 10, 908), 242), fill=(119, 119, 119), width=2)
        d.line((x, 378, min(x + 10, 908), 378), fill=(119, 119, 119), width=2)
        x += 16
    y = 254
    while y < 370:  # dashed left/right
        d.line((662, y, 662, min(y + 10, 366)), fill=(119, 119, 119), width=2)
        d.line((918, y, 918, min(y + 10, 366)), fill=(119, 119, 119), width=2)
        y += 16
    motifs["tint_card_border"] = (960, 240, 1220, 380)
    _rr(d, (960, 240, 1220, 380), 12, fill=(232, 240, 254),
        outline=(66, 133, 244), width=2)
    motifs["hline_solid"] = (60, 440, 360, 443)
    d.line((60, 440, 360, 440), fill=(102, 102, 102), width=3)
    motifs["hline_dashed"] = (400, 440, 700, 442)
    x = 400
    while x < 700:
        d.line((x, 440, min(x + 12, 700), 440), fill=(102, 102, 102), width=2)
        x += 20
    motifs["hline_arrow"] = (740, 435, 1040, 445)
    d.line((740, 440, 1030, 440), fill=(102, 102, 102), width=3)
    d.polygon([(1030, 433), (1044, 440), (1030, 447)], fill=(102, 102, 102))
    motifs["diag_line"] = (60, 520, 360, 640)
    d.line((60, 640, 360, 520), fill=(102, 102, 102), width=3)
    motifs["elbow_line"] = (400, 520, 700, 640)
    d.line((400, 520, 700, 520), fill=(102, 102, 102), width=3)
    d.line((700, 520, 700, 640), fill=(102, 102, 102), width=3)
    motifs["vline"] = (760, 500, 763, 660)
    d.line((760, 500, 760, 660), fill=(102, 102, 102), width=3)
    return img, [], motifs


def _page_shapes_with_text():
    img = _canvas()
    d = ImageDraw.Draw(img)
    ocr = []
    motifs = {}
    motifs["blue_card_text"] = (60, 40, 360, 200)
    _rr(d, (60, 40, 360, 200), 16, fill=(40, 100, 200))
    _draw_text(d, 210, 120, "Plan A", 34, (255, 255, 255), ocr)
    motifs["gray_card_text"] = (420, 40, 720, 200)
    d.rectangle((420, 40, 720, 200), fill=(236, 236, 236))
    _draw_text(d, 570, 120, "Step 1", 34, (51, 51, 51), ocr)
    motifs["pale_card_text"] = (780, 40, 1180, 200)
    _rr(d, (780, 40, 1180, 200), 16, fill=(250, 247, 245))
    _draw_text(d, 980, 120, "Notes", 34, (60, 60, 60), ocr)
    motifs["hollow_card_text"] = (60, 260, 360, 420)
    _rr(d, (60, 260, 360, 420), 16, outline=(85, 85, 85), width=3)
    _draw_text(d, 210, 340, "Title", 30, (60, 60, 60), ocr)
    motifs["badge_circle_text"] = (425, 275, 555, 405)
    d.ellipse((425, 275, 555, 405), fill=(40, 100, 200))
    _draw_text(d, 490, 340, "1", 36, (255, 255, 255), ocr)
    motifs["tint_card_text"] = (780, 260, 1180, 420)
    _rr(d, (780, 260, 1180, 420), 12, fill=(232, 240, 254),
        outline=(66, 133, 244), width=2)
    _draw_text(d, 980, 340, "Data", 30, (40, 60, 120), ocr)
    return img, ocr, motifs


def _page_compositions():
    img = _canvas()
    d = ImageDraw.Draw(img)
    ocr = []
    motifs = {}
    motifs["card_icon_text"] = (60, 60, 420, 240)
    _rr(d, (60, 60, 420, 240), 14, fill=(250, 247, 245))
    d.ellipse((90, 100, 150, 160), fill=(40, 100, 200))
    _draw_text(d, 280, 130, "Metric", 26, (60, 60, 60), ocr)
    motifs["cardA"] = (500, 60, 700, 200)
    d.rounded_rectangle((500, 60, 700, 200), radius=10, fill=(236, 236, 236))
    motifs["cardB"] = (860, 60, 1060, 200)
    d.rounded_rectangle((860, 60, 1060, 200), radius=10, fill=(236, 236, 236))
    motifs["cards_connector"] = (700, 130, 860, 133)
    d.line((700, 130, 860, 130), fill=(120, 120, 120), width=3)
    motifs["header_band_card"] = (60, 300, 420, 520)
    d.rounded_rectangle((60, 300, 420, 520), radius=8, fill=(255, 255, 255),
                        outline=(204, 204, 204), width=2)
    d.rectangle((62, 302, 418, 352), fill=(40, 100, 200))
    _draw_text(d, 240, 327, "Title", 24, (255, 255, 255), ocr)
    _draw_text(d, 240, 430, "Content", 22, (60, 60, 60), ocr)
    motifs["src_circle"] = (500, 320, 580, 400)
    d.ellipse((500, 320, 580, 400), fill=(138, 138, 138))
    motifs["dst_circle"] = (720, 320, 800, 400)
    d.ellipse((720, 320, 800, 400), fill=(138, 138, 138))
    motifs["arrow_link"] = (585, 357, 712, 363)
    d.line((585, 360, 700, 360), fill=(102, 102, 102), width=3)
    d.polygon([(700, 353), (714, 360), (700, 367)], fill=(102, 102, 102))
    for i, x in enumerate((60, 170, 280)):
        motifs[f"chip_{i}"] = (x, 560, x + 90, 596)
        _rr(d, (x, 560, x + 90, 596), 18, fill=(40, 100, 200))
    return img, ocr, motifs


def _overlap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ox * oy
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(aa, ba)


def _elements_for(layout: dict, motif_box: tuple) -> list[dict]:
    hits = []
    for el in layout["elements"]:
        eb = el.get("box")
        if not eb:
            continue
        ex1, ey1 = eb[0], eb[1]
        ex2, ey2 = eb[0] + eb[2], eb[1] + eb[3]
        if el.get("type") == "line":
            pts = el["points"]
            ex1, ey1 = min(pts[0::2]) - 4, min(pts[1::2]) - 4
            ex2, ey2 = max(pts[0::2]) + 4, max(pts[1::2]) + 4
        if _overlap((ex1, ey1, ex2, ey2), motif_box) > 0.3:
            hits.append(el)
    return hits


class _MotifRunner:
    """Build the three pages once and expose the resulting layouts."""

    def __init__(self, base: Path):
        import run_pipeline as rp
        self.base = base
        src = base / "src"
        work = base / "work"
        (work / "ocr").mkdir(parents=True, exist_ok=True)
        src.mkdir(parents=True, exist_ok=True)
        self.pages = {}
        for num, gen in (("01", _page_shapes_plain),
                         ("02", _page_shapes_with_text),
                         ("03", _page_compositions)):
            img, ocr, motifs = gen()
            img.save(src / f"page_{num}.png")
            (work / "ocr" / f"page_{num}.ocr.json").write_text(
                json.dumps(ocr, ensure_ascii=False), encoding="utf-8")
            rp.process_page(num, src, work)
            layout = json.loads(
                (work / "layouts" / f"page_{num}.layout.json")
                .read_text(encoding="utf-8"))
            self.pages[num] = (motifs, layout)


class MotifPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.runner = _MotifRunner(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _hits(self, page: str, motif: str) -> list[dict]:
        motifs, layout = self.runner.pages[page]
        return _elements_for(layout, motifs[motif])

    def _kinds(self, page: str, motif: str) -> list[str]:
        return [el["type"] for el in self._hits(page, motif)]

    # ------------------------------------------------------------ page 1
    def test_plain_shapes_lift_native(self) -> None:
        for motif in ("filled_round_rect", "filled_rect", "pale_card",
                      "circle", "hollow_rect", "hollow_round_rect",
                      "tint_card_border"):
            with self.subTest(motif=motif):
                self.assertIn("shape", self._kinds("01", motif),
                              f"{motif} did not lift to a native shape")
                self.assertNotIn("image", self._kinds("01", motif))

    def test_dashed_frame_not_flattened(self) -> None:
        """A dashed frame must stay native: either one dashed_* shape, or
        its edges decomposed into dashed lines (a corner-gap dash style
        bridging into four connectors). A flattened PNG loses the dash
        styling."""
        hits = self._hits("01", "dashed_frame")
        kinds = [e["type"] for e in hits]
        self.assertNotIn("image", kinds)
        dashed_shapes = [e for e in hits if e["type"] == "shape"
                         and str(e.get("shape", "")).startswith("dashed")]
        dashed_lines = [e for e in hits if e["type"] == "line"
                        and e.get("dash")]
        self.assertTrue(dashed_shapes or len(dashed_lines) >= 2,
                        f"dashed frame lost its dash styling: {kinds}")

    def test_plain_lines_lift_native(self) -> None:
        for motif in ("hline_solid", "hline_dashed", "hline_arrow",
                      "diag_line", "elbow_line", "vline"):
            with self.subTest(motif=motif):
                self.assertIn("line", self._kinds("01", motif),
                              f"{motif} did not lift to a native line")
                self.assertNotIn("image", self._kinds("01", motif))

    def test_dashed_line_carries_dash(self) -> None:
        lines = [e for e in self._hits("01", "hline_dashed")
                 if e["type"] == "line"]
        self.assertTrue(lines)
        self.assertEqual(lines[0].get("dash"), "dash")

    def test_arrow_line_carries_arrow(self) -> None:
        lines = [e for e in self._hits("01", "hline_arrow")
                 if e["type"] == "line"]
        self.assertTrue(lines)
        self.assertEqual(lines[0].get("arrow"), "end")

    # ------------------------------------------------------------ page 2
    def test_cards_with_text_lift_native(self) -> None:
        # pale_card_text is the regression this matrix exists for: the
        # same card without text lifts, and erase remnants used to make
        # the with-text variant collapse to a flattened PNG.
        for motif in ("blue_card_text", "gray_card_text", "pale_card_text",
                      "hollow_card_text", "badge_circle_text",
                      "tint_card_text"):
            with self.subTest(motif=motif):
                kinds = self._kinds("02", motif)
                self.assertIn("shape", kinds,
                              f"{motif} did not lift to a native shape")
                self.assertNotIn("image", kinds)

    def test_badge_circle_lifts_oval(self) -> None:
        shapes = [e for e in self._hits("02", "badge_circle_text")
                  if e["type"] == "shape"]
        self.assertTrue(any(s.get("shape") == "oval" for s in shapes))

    # ------------------------------------------------------------ page 3
    def test_card_with_icon_and_text_decomposes(self) -> None:
        kinds = self._kinds("03", "card_icon_text")
        self.assertIn("shape", kinds)
        self.assertNotIn("image", kinds)

    def test_connected_cards_decompose(self) -> None:
        """The flow-diagram pattern: two cards + connector glued into one
        connected component must still yield native members."""
        for motif in ("cardA", "cardB"):
            with self.subTest(motif=motif):
                self.assertIn("shape", self._kinds("03", motif))
                self.assertNotIn("image", self._kinds("03", motif))
        self.assertIn("line", self._kinds("03", "cards_connector"))
        self.assertNotIn("image", self._kinds("03", "cards_connector"))

    def test_header_band_lifts_native(self) -> None:
        hits = self._hits("03", "header_band_card")
        self.assertTrue(any(e["type"] == "shape" for e in hits))

    def test_circles_and_arrow_no_double_render(self) -> None:
        for motif in ("src_circle", "dst_circle"):
            with self.subTest(motif=motif):
                self.assertIn("shape", self._kinds("03", motif))
                self.assertNotIn("image", self._kinds("03", motif))
        kinds = self._kinds("03", "arrow_link")
        self.assertIn("line", kinds)
        self.assertNotIn("image", kinds)

    def test_chips_lift_native(self) -> None:
        for i in range(3):
            motif = f"chip_{i}"
            with self.subTest(motif=motif):
                self.assertIn("shape", self._kinds("03", motif))
                self.assertNotIn("image", self._kinds("03", motif))


if __name__ == "__main__":
    unittest.main()
