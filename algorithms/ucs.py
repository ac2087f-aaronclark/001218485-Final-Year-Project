from __future__ import annotations

# Implements Uniform Cost Search for the weighted grid.
# It expands nodes using cumulative path cost only.

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from maze.grid import Grid, Pos

# Algorithm code credit:
# https://github.com/adityadas8888/UniformCostSearch/blob/master/find_route.py
# The code was then adapted for A* and Weighted A* in this project.


# Stores the outputs from one Uniform Cost Search run.
@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
    best_cost: Dict[Pos, float]          # Best known cost from start to each visited cell
    parent: Dict[Pos, Optional[Pos]]     # Parent links used to rebuild the final path


# Rebuilds the path from goal back to start using the parent map.
def reconstruct_path(parent: Dict[Pos, Optional[Pos]], start: Pos, goal: Pos) -> List[Pos]:
    # If the goal was never reached then no path exists.
    if goal not in parent:
        return []

    path: List[Pos] = []
    cur: Optional[Pos] = goal

    # Follow parent links backward from goal to start.
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)

    # Reverse the backward path so it runs from start to goal.
    path.reverse()

    # Safety check to make sure the reconstructed path starts correctly.
    if not path or path[0] != start:
        return []

    return path


# Runs Uniform Cost Search from the start cell to the goal cell.
def uniform_cost_search(grid: Grid, start: Pos, goal: Pos) -> SearchResult:
    # Priority queue stores (path cost so far, node).
    pq: List[Tuple[float, Pos]] = []
    heapq.heappush(pq, (0.0, start))

    # best_cost stores the cheapest known route from the start to each node.
    best_cost: Dict[Pos, float] = {start: 0.0}

    # parent stores how each node was reached so the final path can be rebuilt.
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    # Counts how many nodes are actually expanded.
    nodes_expanded = 0

    while pq:
        cost, node = heapq.heappop(pq)

        # Ignore stale queue entries if a cheaper route was found later.
        if cost != best_cost.get(node, float("inf")):
            continue

        nodes_expanded += 1

        # Stop once the goal is selected for expansion.
        if node == goal:
            break

        # Check each valid neighbour and relax the edge if a cheaper route is found.
        for nbr in grid.neighbors(node):
            step = grid.step_cost(nbr)

            # UCS requires non negative movement costs.
            if step < 0:
                raise ValueError("Negative cost detected (UCS/Dijkstra requires non-negative costs).")

            new_cost = cost + float(step)

            # Update the neighbour only if this route improves its best known cost.
            if new_cost < best_cost.get(nbr, float("inf")):
                best_cost[nbr] = new_cost
                parent[nbr] = node
                heapq.heappush(pq, (new_cost, nbr))

    # Build the final path and read the total cost to the goal.
    path = reconstruct_path(parent, start, goal)
    total = best_cost.get(goal, float("inf"))

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        best_cost=best_cost,
        parent=parent,
    )