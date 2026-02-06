from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import heapq

from maze.grid import Grid, Pos

#algorithm code credit https://github.com/adityadas8888/UniformCostSearch/blob/master/find_route.py
#code was then adapted to A_hash and Weighted_A_hash

@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
    best_cost: Dict[Pos, float]
    parent: Dict[Pos, Optional[Pos]]


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


def uniform_cost_search(grid: Grid, start: Pos, goal: Pos) -> SearchResult:
    """
    Uniform Cost Search on your grid (equivalent to Dijkstra for non-negative costs).

    - Node/state: (row, col)
    - Edge cost: cost to ENTER neighbour cell = grid.step_cost(neighbour)
    """

    pq: List[Tuple[float, Pos]] = []
    heapq.heappush(pq, (0.0, start))

    best_cost: Dict[Pos, float] = {start: 0.0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    nodes_expanded = 0

    while pq:
        cost, node = heapq.heappop(pq)

        # Skip stale entries
        if cost != best_cost.get(node, float("inf")):
            continue

        nodes_expanded += 1

        if node == goal:
            break

        for nbr in grid.neighbors(node):
            step = grid.step_cost(nbr)
            if step < 0:
                raise ValueError("Negative cost detected (UCS/Dijkstra requires non-negative costs).")

            new_cost = cost + float(step)

            if new_cost < best_cost.get(nbr, float("inf")):
                best_cost[nbr] = new_cost
                parent[nbr] = node
                heapq.heappush(pq, (new_cost, nbr))

    path = reconstruct_path(parent, start, goal)
    total = best_cost.get(goal, float("inf"))

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        best_cost=best_cost,
        parent=parent,
    )