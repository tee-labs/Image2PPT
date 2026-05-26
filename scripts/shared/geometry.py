"""Shared bbox / colour geometry helpers.

Replaces three identical `_intersection_area` / `_box_intersection`
definitions across run_pipeline / build_inventory / inventory_to_layout,
plus the duplicated `_connector_on_container_border`, `_overlaps_text`,
and `_bgr_to_hex` helpers.
"""
from __future__ import annotations


Box = tuple[int, int, int, int]


def intersection_area(a: Box, b: Box) -> int:
    """Pixel area of overlap between two (x1, y1, x2, y2) boxes. 0 if disjoint."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ox1, oy1 = max(ax1, bx1), max(ay1, by1)
    ox2, oy2 = min(ax2, bx2), min(ay2, by2)
    if ox2 <= ox1 or oy2 <= oy1:
        return 0
    return (ox2 - ox1) * (oy2 - oy1)


def overlaps_text(box: Box, text_boxes: list[Box]) -> bool:
    return any(intersection_area(box, tb) > 0 for tb in text_boxes)


def bgr_to_hex(color) -> str:
    """`np.ndarray | list | tuple` of BGR → `#RRGGBB`."""
    b, g, r = (int(v) for v in color[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def connector_on_container_border(container_box: Box,
                                  connector_box: Box) -> bool:
    """Return true when a connector bbox is really a container border dash.

    Used to suppress connector elements that are visually part of a card's
    own dashed/dotted frame.
    """
    px1, py1, px2, py2 = container_box
    cx1, cy1, cx2, cy2 = connector_box
    c_area = max(1, (cx2 - cx1) * (cy2 - cy1))
    if intersection_area(container_box, connector_box) < 0.80 * c_area:
        return False
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    cw = max(1, cx2 - cx1)
    ch = max(1, cy2 - cy1)
    band = max(4, min(14, int(round(min(pw, ph) * 0.22))))
    near_top = cy1 - py1 <= band
    near_bottom = py2 - cy2 <= band
    near_left = cx1 - px1 <= band
    near_right = px2 - cx2 <= band
    horizontal = cw >= max(12, ch * 2.0)
    vertical = ch >= max(12, cw * 2.0)
    if horizontal and (near_top or near_bottom):
        return True
    if vertical and (near_left or near_right):
        return True
    return bool((near_top or near_bottom) and (near_left or near_right))
