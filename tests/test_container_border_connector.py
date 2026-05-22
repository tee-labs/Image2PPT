from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "scripts" / "page"
if str(PAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PAGE_DIR))

from build_inventory import (  # noqa: E402
    _connector_on_container_border as inventory_connector_on_container_border,
)
from inventory_to_layout import (  # noqa: E402
    _connector_on_container_border as layout_connector_on_container_border,
)


class ContainerBorderConnectorTests(unittest.TestCase):
    def test_connector_on_container_edge_is_border_dash(self) -> None:
        self.assertTrue(
            layout_connector_on_container_border(
                (909, 173, 1227, 225),
                (915, 213, 962, 221),
            )
        )
        self.assertTrue(
            inventory_connector_on_container_border(
                (909, 173, 1227, 225),
                (915, 213, 962, 221),
            )
        )

    def test_connector_inside_container_stays_independent(self) -> None:
        self.assertFalse(
            layout_connector_on_container_border(
                (909, 173, 1227, 225),
                (980, 193, 1040, 201),
            )
        )
        self.assertFalse(
            inventory_connector_on_container_border(
                (909, 173, 1227, 225),
                (980, 193, 1040, 201),
            )
        )


if __name__ == "__main__":
    unittest.main()
