# algorithms/astar.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import heapq

from maze.grid import Grid, Pos


@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
    best_cost: Dict[Pos, float]          # g-score
    parent: Dict[Pos, Optional[Pos]]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(parent: Dict[Pos, Optional[Pos]], start: Pos, goal: Pos) -> List[Pos]:
    if goal not in parent:
        return []

    path: List[Pos] = []
    cur: Optional[Pos] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    if not path or path[0] != start:
        return []
    return path


def a_star_search(grid: Grid, start: Pos, goal: Pos) -> SearchResult:
    """
    A* on your grid (4-connected).
    - g(n): cost so far (sum of grid.step_cost into cells)
    - h(n): Manhattan distance to goal
    - f(n) = g(n) + h(n)
    """

    pq: List[Tuple[float, Pos]] = []
    g_score: Dict[Pos, float] = {start: 0.0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    f0 = 0.0 + float(manhattan(start, goal))
    heapq.heappush(pq, (f0, start))

    nodes_expanded = 0

    while pq:
        f, node = heapq.heappop(pq)

        # Skip stale entries (based on the best-known f for this node)
        expected_f = g_score.get(node, float("inf")) + float(manhattan(node, goal))
        if f != expected_f:
            continue

        nodes_expanded += 1

        if node == goal:
            break

        current_g = g_score[node]

        for nbr in grid.neighbors(node):
            step = grid.step_cost(nbr)
            if step < 0:
                raise ValueError("Negative cost detected (A* requires non-negative costs).")

            new_g = current_g + float(step)

            if new_g < g_score.get(nbr, float("inf")):
                g_score[nbr] = new_g
                parent[nbr] = node

                new_f = new_g + float(manhattan(nbr, goal))
                heapq.heappush(pq, (new_f, nbr))

    path = reconstruct_path(parent, start, goal)
    total = g_score.get(goal, float("inf"))

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        best_cost=g_score,
        parent=parent,
    )
