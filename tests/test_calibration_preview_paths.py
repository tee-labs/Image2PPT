from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from calibrate_text_positions import _preview_for_slide  # noqa: E402


class CalibrationPreviewPathTests(unittest.TestCase):
    def test_finds_zero_padded_preview_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preview_dir = Path(tmp)
            expected = preview_dir / "page-04.png"
            expected.write_bytes(b"")

            self.assertEqual(_preview_for_slide(preview_dir, 3), expected)

    def test_prefers_non_padded_preview_names_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            preview_dir = Path(tmp)
            unpadded = preview_dir / "page-4.png"
            padded = preview_dir / "page-04.png"
            unpadded.write_bytes(b"")
            padded.write_bytes(b"")

            self.assertEqual(_preview_for_slide(preview_dir, 3), unpadded)


if __name__ == "__main__":
    unittest.main()
