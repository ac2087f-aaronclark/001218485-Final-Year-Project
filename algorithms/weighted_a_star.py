from __future__ import annotations

"""
This file implements Weighted A* search for the project.

It searches the weighted 4-connected grid using path cost so far plus
a weighted Manhattan-distance heuristic, and returns the path and run metrics.
"""

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from maze.grid import Grid, Pos


# Stores the output of one Weighted A* search run.
@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
    best_cost: Dict[Pos, float]          # g-score
    parent: Dict[Pos, Optional[Pos]]
    weight: float


# Returns the Manhattan-distance heuristic between two grid cells.
def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Reconstructs the path from the parent map if the goal was reached.
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


# Runs Weighted A* search on the grid from the start cell to the goal cell.
def weighted_a_star_search(grid: Grid, start: Pos, goal: Pos, w: float = 1.5) -> SearchResult:
    """
    Weighted A* on your grid (4-connected).
    - g(n): cost so far
    - h(n): Manhattan distance
    - f(n) = g(n) + w*h(n)

    Notes:
    - w = 1.0 -> standard A*
    - w > 1.0 -> more greedy / faster, may be suboptimal
    """
    if w <= 0:
        raise ValueError("w must be > 0")

    pq: List[Tuple[float, Pos]] = []
    g_score: Dict[Pos, float] = {start: 0.0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    f0 = 0.0 + w * float(manhattan(start, goal))
    heapq.heappush(pq, (f0, start))

    nodes_expanded = 0

    while pq:
        f, node = heapq.heappop(pq)

        # Skip stale queue entries using the current best-known weighted f-score.
        expected_f = g_score.get(node, float("inf")) + w * float(manhattan(node, goal))
        if f != expected_f:
            continue

        nodes_expanded += 1

        if node == goal:
            break

        current_g = g_score[node]

        # Relax each valid neighbour of the current node.
        for nbr in grid.neighbors(node):
            step = grid.step_cost(nbr)
            if step < 0:
                raise ValueError("Negative cost detected (Weighted A* requires non-negative costs).")

            new_g = current_g + float(step)

            if new_g < g_score.get(nbr, float("inf")):
                g_score[nbr] = new_g
                parent[nbr] = node

                new_f = new_g + w * float(manhattan(nbr, goal))
                heapq.heappush(pq, (new_f, nbr))

    path = reconstruct_path(parent, start, goal)
    total = g_score.get(goal, float("inf"))

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        best_cost=g_score,
        parent=parent,
        weight=w,
    )