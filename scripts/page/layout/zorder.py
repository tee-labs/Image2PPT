"""Topologically order image elements by bbox containment.

Parents must render before their children so a card never paints over an
icon nested inside it.
"""
from __future__ import annotations


def topo_sort_by_containment(
    elements: list[dict],
    containment: float = 0.85,
) -> list[dict]:
    """Reorder image elements so visual parents render before their children.

    Builds a DAG where `parent → child` whenever `parent.bbox` is strictly
    larger than `child.bbox` AND covers at least `containment` (default
    85 %) of the child's area. A Kahn-style topological sort then emits
    every parent before any of its descendants, so PowerPoint paints
    parents first (back) and children last (front). Without this, a
    parent card can land *above* an icon nested inside it and hide it.

    The strict `parent_area > child_area` requirement prevents cycles
    between two roughly equal-size overlapping boxes (neither parents
    the other); the threshold prevents incidental edge-touches from
    creating spurious parent links.

    Tiebreaker for nodes with no remaining parents: larger area first
    (the natural back), then top-to-bottom, left-to-right. Falls back to
    the input order if a cycle is somehow produced (shouldn't happen
    given the strict-area rule, but the guard is cheap).
    """
    n = len(elements)
    if n < 2:
        return elements

    boxes = []
    for e in elements:
        x, y, w, h = e["box"]
        boxes.append((x, y, x + w, y + h, max(1, w * h)))

    parents_of: list[set[int]] = [set() for _ in range(n)]
    children_of: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        ix1, iy1, ix2, iy2, ia = boxes[i]
        for j in range(n):
            if i == j:
                continue
            jx1, jy1, jx2, jy2, ja = boxes[j]
            if ja <= ia:
                continue
            ox1 = max(ix1, jx1)
            oy1 = max(iy1, jy1)
            ox2 = min(ix2, jx2)
            oy2 = min(iy2, jy2)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            inter = (ox2 - ox1) * (oy2 - oy1)
            if inter >= containment * ia:
                parents_of[i].add(j)
                children_of[j].add(i)

    def tiebreak(i: int) -> tuple:
        x, y, _, _, area = boxes[i]
        return (-area, y, x, i)

    indeg = [len(p) for p in parents_of]
    available = sorted([i for i in range(n) if indeg[i] == 0], key=tiebreak)

    result: list[dict] = []
    while available:
        i = available.pop(0)
        result.append(elements[i])
        for c in children_of[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ck = tiebreak(c)
                lo = 0
                hi = len(available)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tiebreak(available[mid]) <= ck:
                        lo = mid + 1
                    else:
                        hi = mid
                available.insert(lo, c)

    if len(result) != n:
        return elements
    return result
