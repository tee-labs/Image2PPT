from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from classify_text_slots import classify_slide  # noqa: E402


class TextSizeClassificationThresholdTests(unittest.TestCase):
    def test_size_delta_over_two_does_not_merge_title_and_subtitle(self) -> None:
        slide = {
            "source_width": 800,
            "source_height": 450,
            "elements": [
                {"type": "image", "name": "card", "box": [40, 40, 720, 240]},
                {
                    "type": "text",
                    "name": "cn_left",
                    "text": "系统设计与落地",
                    "box": [100, 90, 160, 30],
                    "color": "#054798",
                    "size": 20,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "cn_right",
                    "text": "攻防安全",
                    "box": [420, 90, 120, 30],
                    "color": "#054798",
                    "size": 20,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "en_left",
                    "text": "System Engineering",
                    "box": [100, 122, 180, 24],
                    "color": "#054798",
                    "size": 16,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "en_right",
                    "text": "Security Analysis",
                    "box": [420, 122, 180, 24],
                    "color": "#054798",
                    "size": 16,
                    "align": "left",
                },
            ],
        }

        report = classify_slide(slide, min_group_size=2, apply=True)

        by_name = {
            element["name"]: element
            for element in slide["elements"]
            if element.get("type") == "text"
        }
        self.assertEqual(by_name["cn_left"]["size"], 20)
        self.assertEqual(by_name["cn_right"]["size"], 20)
        self.assertEqual(by_name["en_left"]["size"], 16)
        self.assertEqual(by_name["en_right"]["size"], 16)

        class_members = [
            {member["name"] for member in cls["members"]}
            for cls in report["classes"]
        ]
        self.assertIn({"cn_left", "cn_right"}, class_members)
        self.assertIn({"en_left", "en_right"}, class_members)
        self.assertNotIn(
            {"cn_left", "cn_right", "en_left", "en_right"},
            class_members,
        )

    def test_compact_latin_card_titles_share_slot_without_size_normalization(self) -> None:
        slide = {
            "source_width": 1280,
            "source_height": 720,
            "elements": [
                {
                    "type": "image",
                    "name": "overview_cards",
                    "box": [46, 107, 1207, 524],
                },
                {
                    "type": "text",
                    "name": "title_ents",
                    "text": "Ents",
                    "box": [259, 334, 146, 70],
                    "color": "#054798",
                    "font": "Microsoft YaHei",
                    "size": 36,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "title_guard",
                    "text": "MPCGuard",
                    "box": [636, 356, 214, 52],
                    "color": "#054798",
                    "font": "Microsoft YaHei",
                    "size": 26,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "title_arbiter",
                    "text": "MPCArbiter",
                    "box": [1002, 357, 274, 54],
                    "color": "#054798",
                    "font": "Microsoft YaHei",
                    "size": 26,
                    "bold": True,
                    "align": "left",
                },
            ],
        }

        report = classify_slide(slide, min_group_size=2, apply=True)

        titles = [
            element for element in slide["elements"]
            if str(element.get("name", "")).startswith("title_")
        ]
        classes = {element.get("style_class") for element in titles}
        self.assertEqual(len(classes), 1)
        self.assertNotIn(None, classes)
        self.assertEqual(
            {element["name"]: element["size"] for element in titles},
            {
                "title_ents": 36,
                "title_guard": 26,
                "title_arbiter": 26,
            },
        )
        self.assertTrue(
            all("style_class_suggested_size" not in element for element in titles)
        )
        title_class = next(
            cls for cls in report["classes"]
            if cls["class_id"] == next(iter(classes))
        )
        self.assertTrue(title_class["size_exception"])
        self.assertFalse(title_class["size_unification_eligible"])
        self.assertIsNone(title_class["suggested_size"])
        edge_reasons = [
            reason
            for _a, _b, reasons in title_class["edge_evidence"]
            for reason in reasons
        ]
        self.assertIn("compact_latin_title", edge_reasons)

    def test_stacked_toc_items_share_slot_and_unify_close_sizes(self) -> None:
        slide = {
            "source_width": 1280,
            "source_height": 720,
            "elements": [
                {"type": "image", "name": "band_1", "box": [312, 138, 698, 73]},
                {"type": "image", "name": "band_2", "box": [312, 232, 698, 72]},
                {"type": "image", "name": "band_3", "box": [312, 325, 698, 73]},
                {"type": "image", "name": "lower_group", "box": [311, 299, 969, 421]},
                {"type": "image", "name": "right_overlay", "box": [704, 299, 575, 411]},
                {
                    "type": "text",
                    "name": "toc_intro",
                    "text": "绪论",
                    "box": [340.12, 152.08, 105, 46],
                    "color": "#054798",
                    "size": 25,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "toc_ents",
                    "text": "基于MPC的高效树模型训练框架Ents",
                    "box": [340.12, 240.72, 613, 49],
                    "color": "#054798",
                    "size": 23,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "toc_guard",
                    "text": "MPC协议实现数据泄露漏洞检测框架 MPCGuard",
                    "box": [351.44, 337.64, 755, 42],
                    "color": "#054798",
                    "size": 23,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "toc_arbiter",
                    "text": "MPC协议实现数值错误漏洞检测框架 MPCArbiter",
                    "box": [354.92, 427.28, 796, 50],
                    "color": "#054798",
                    "size": 25,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "toc_summary",
                    "text": "总结与展望",
                    "box": [340.12, 526.44, 190, 44],
                    "color": "#054798",
                    "size": 23,
                    "bold": True,
                    "align": "left",
                },
            ],
        }

        report = classify_slide(slide, min_group_size=2, apply=True)

        items = [
            element for element in slide["elements"]
            if str(element.get("name", "")).startswith("toc_")
        ]
        classes = {element.get("style_class") for element in items}
        self.assertEqual(len(classes), 1)
        self.assertNotIn(None, classes)
        self.assertEqual({element["size"] for element in items}, {23})
        self.assertEqual(
            {element.get("style_class_suggested_size") for element in items},
            {23},
        )
        toc_class = next(
            cls for cls in report["classes"]
            if cls["class_id"] == next(iter(classes))
        )
        self.assertTrue(toc_class["size_unification_eligible"])
        edge_reasons = [
            reason
            for _a, _b, reasons in toc_class["edge_evidence"]
            for reason in reasons
        ]
        self.assertIn("vertical_list_slot", edge_reasons)


if __name__ == "__main__":
    unittest.main()
