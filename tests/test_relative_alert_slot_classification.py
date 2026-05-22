from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "scripts" / "deck"
if str(DECK_DIR) not in sys.path:
    sys.path.insert(0, str(DECK_DIR))

from classify_text_slots import classify_slide  # noqa: E402


class RelativeAlertSlotClassificationTests(unittest.TestCase):
    def test_nested_card_warning_text_matches_sibling_card(self) -> None:
        slide = {
            "source_width": 1280,
            "source_height": 720,
            "elements": [
                {
                    "type": "image",
                    "name": "left_outer_card",
                    "box": [42, 116, 584, 421],
                },
                {
                    "type": "image",
                    "name": "left_inner_content",
                    "box": [45, 214, 576, 319],
                },
                {
                    "type": "image",
                    "name": "right_card",
                    "box": [646, 116, 592, 421],
                },
                {
                    "type": "text",
                    "name": "right_warning_1",
                    "text": "主要侧重模型精度比对或编译器逻辑正确性一",
                    "box": [737, 430, 515, 30],
                    "color": "#D43E3E",
                    "size": 16,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "left_warning_1",
                    "text": "关注协议逻辑的密码学健全性，",
                    "box": [165.48, 445.28, 341, 30],
                    "color": "#D43E3E",
                    "size": 15,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "right_warning_2",
                    "text": "缺乏对通用MPC协议实现中数据泄露与数值错误",
                    "box": [737, 459.64, 532, 32],
                    "color": "#D43E3E",
                    "size": 16,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "left_warning_2",
                    "text": "抽象掉实际软件实现细节一无法发现实现层面漏洞",
                    "box": [140.32, 476, 566, 30],
                    "color": "#D43E3E",
                    "size": 16,
                    "bold": True,
                    "align": "left",
                },
                {
                    "type": "text",
                    "name": "right_warning_3",
                    "text": "漏洞的系统化检测",
                    "box": [737, 490.2, 211, 30],
                    "color": "#D43E3E",
                    "size": 16,
                    "bold": True,
                    "align": "left",
                },
            ],
        }

        report = classify_slide(slide, min_group_size=2, apply=True)

        warnings = [
            element for element in slide["elements"]
            if str(element.get("name", "")).endswith("warning_1")
            or str(element.get("name", "")).endswith("warning_2")
            or str(element.get("name", "")).endswith("warning_3")
        ]
        classes = {element.get("style_class") for element in warnings}
        self.assertEqual(len(classes), 1)
        self.assertNotIn(None, classes)
        self.assertEqual({element.get("size") for element in warnings}, {16})

        warning_class = next(
            cls for cls in report["classes"]
            if cls["class_id"] == next(iter(classes))
        )
        edge_reasons = [
            reason
            for _a, _b, reasons in warning_class["edge_evidence"]
            for reason in reasons
        ]
        self.assertIn("relative_alert_band", edge_reasons)


if __name__ == "__main__":
    unittest.main()
