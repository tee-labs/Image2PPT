from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from build_pptx_from_layout import (  # noqa: E402
    apply_class_alignment,
    apply_title_centering,
)


class TitleCenteringPriorityTests(unittest.TestCase):
    def test_title_centering_runs_before_classified_alignment(self) -> None:
        elements = [
            {"type": "image", "box": [0, 0, 100, 100]},
            {
                "type": "text",
                "text": "标题",
                "box": [42, 40, 10, 10],
                "align": "left",
                "style_class_align": "left",
            },
        ]

        apply_title_centering(elements)
        apply_class_alignment(elements)

        self.assertEqual(elements[1]["box"], [45, 40, 10, 10])
        self.assertEqual(elements[1]["align"], "left")

    def test_title_centering_still_applies_without_classification(self) -> None:
        elements = [
            {"type": "image", "box": [0, 0, 100, 100]},
            {"type": "text", "text": "标题", "box": [42, 40, 10, 10],
             "align": "left"},
        ]

        apply_title_centering(elements)

        self.assertEqual(elements[1]["box"], [45, 40, 10, 10])
        self.assertEqual(elements[1]["align"], "center")

    def test_class_alignment_is_final_override(self) -> None:
        elements = [
            {
                "type": "text",
                "text": "正文",
                "box": [10, 10, 40, 12],
                "align": "center",
                "style_class_align": "left",
            },
            {
                "type": "text",
                "text": "说明",
                "box": [10, 30, 40, 12],
                "align": "center",
                "style_parent_column_align": "right",
            },
        ]

        apply_class_alignment(elements)

        self.assertEqual(elements[0]["align"], "left")
        self.assertEqual(elements[1]["align"], "right")


if __name__ == "__main__":
    unittest.main()
