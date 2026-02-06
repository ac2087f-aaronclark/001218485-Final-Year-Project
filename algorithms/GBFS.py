# algorithms/greedy_best_first.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import heapq

from maze.grid import Grid, Pos

#algorithm code credit goes to https://github.com/shai-raz/AStar-GreedyBestFirstSearch-Pathfinder-with-GUI/blob/master/alogs.py

@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
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


def path_cost(grid: Grid, path: List[Pos]) -> float:
    """
    Total cost for a path using the same convention as UCS/A*:
    cost to ENTER each cell (excluding the start cell).
    """
    if not path:
        return float("inf")

    total = 0.0
    for i in range(1, len(path)):
        total += float(grid.step_cost(path[i]))
    return total


def greedy_best_first_search(grid: Grid, start: Pos, goal: Pos) -> SearchResult:
    """
    Greedy Best-First Search (GBFS):
    - Priority is ONLY the heuristic h(n)
    - Not optimal (may return higher-cost paths)
    """

    pq: List[Tuple[float, Pos]] = []
    heapq.heappush(pq, (float(manhattan(start, goal)), start))

    parent: Dict[Pos, Optional[Pos]] = {start: None}
    visited: set[Pos] = set()

    nodes_expanded = 0

    while pq:
        _, node = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        nodes_expanded += 1

        if node == goal:
            break

        for nbr in grid.neighbors(node):
            if nbr in visited:
                continue

            # Only set parent the first time we discover the node.
            # (Greedy doesn't have a cost-based improvement rule like A*/Dijkstra.)
            if nbr not in parent:
                parent[nbr] = node

            heapq.heappush(pq, (float(manhattan(nbr, goal)), nbr))

    path = reconstruct_path(parent, start, goal)
    total = path_cost(grid, path)

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        parent=parent,
    )
