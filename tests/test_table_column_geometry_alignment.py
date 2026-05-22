from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from classify_text_slots import classify_slide, _table_column_target_x  # noqa: E402


class TableColumnGeometryAlignmentTests(unittest.TestCase):
    def test_aligns_short_table_column_across_colours(self) -> None:
        slide = {
            "source_width": 640,
            "source_height": 480,
            "elements": [
                {"type": "image", "name": "table", "box": [40, 40, 560, 360]},
                {
                    "type": "text",
                    "name": "role_1",
                    "text": "第一作者",
                    "box": [101, 95, 72, 24],
                    "color": "#222222",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "title_1",
                    "text": "Long paper title one",
                    "box": [230, 95, 230, 24],
                    "color": "#222222",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "role_2",
                    "text": "共同一作",
                    "box": [100, 145, 76, 24],
                    "color": "#878789",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "title_2",
                    "text": "Long paper title two",
                    "box": [230, 145, 230, 24],
                    "color": "#222222",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "role_3",
                    "text": "第三作者",
                    "box": [82, 195, 82, 24],
                    "color": "#054798",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "title_3",
                    "text": "Long paper title three",
                    "box": [230, 195, 240, 24],
                    "color": "#222222",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "role_4",
                    "text": "第五作者",
                    "box": [99, 245, 76, 24],
                    "color": "#D43E3E",
                    "size": 13,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "title_4",
                    "text": "Long paper title four",
                    "box": [230, 245, 230, 24],
                    "color": "#222222",
                    "size": 13,
                    "align": "left",
                },
            ],
        }

        report = classify_slide(slide, min_group_size=2, apply=True)

        roles = {
            element["name"]: element
            for element in slide["elements"]
            if element.get("name", "").startswith("role_")
        }
        self.assertEqual(roles["role_1"]["box"][0], 100)
        self.assertEqual(roles["role_2"]["box"][0], 100)
        self.assertEqual(roles["role_3"]["box"][0], 100)
        self.assertEqual(roles["role_4"]["box"][0], 100)
        self.assertTrue(report["table_column_alignment"])
        moved = report["table_column_alignment"][0]["members"]
        self.assertEqual(
            {item["name"] for item in moved},
            {"role_1", "role_3", "role_4"},
        )

    def test_does_not_align_short_labels_without_vertical_rows(self) -> None:
        elements = [
            {"type": "image", "name": "band", "box": [40, 40, 560, 120]},
        ]
        elements.extend(
            {
                "type": "text",
                "name": f"label_{idx}",
                "text": f"L{idx}",
                "box": [80 + idx * 40, 80, 30, 20],
                "color": "#222222",
                "size": 12,
                "align": "left",
            }
            for idx in range(6)
        )
        slide = {
            "source_width": 640,
            "source_height": 480,
            "elements": elements,
        }

        report = classify_slide(slide, min_group_size=2, apply=True)

        xs = [
            element["box"][0]
            for element in slide["elements"]
            if str(element.get("name", "")).startswith("label_")
        ]
        self.assertEqual(xs, [80, 120, 160, 200, 240, 280])
        self.assertEqual(report["table_column_alignment"], [])

    def test_no_table_column_target_for_all_singleton_xs(self) -> None:
        items = [
            {"el": {"box": [80, 50, 40, 20]}},
            {"el": {"box": [110, 90, 40, 20]}},
            {"el": {"box": [150, 130, 40, 20]}},
        ]

        self.assertIsNone(_table_column_target_x(items))


if __name__ == "__main__":
    unittest.main()
