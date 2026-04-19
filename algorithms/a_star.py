from __future__ import annotations

# Implements A* search for the weighted 4-connected grid.
# The algorithm combines cost so far with Manhattan distance to the goal.

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from maze.grid import Grid, Pos


# Stores the main outputs from one A* run.
@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
    best_cost: Dict[Pos, float]          # Best known g-score for each visited cell
    parent: Dict[Pos, Optional[Pos]]     # Parent links used to rebuild the final path


# Returns Manhattan distance between two grid positions.
def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Rebuilds the path by following parent links backward from goal to start.
def reconstruct_path(parent: Dict[Pos, Optional[Pos]], start: Pos, goal: Pos) -> List[Pos]:
    # If the goal was never reached, no path exists.
    if goal not in parent:
        return []

    path: List[Pos] = []
    cur: Optional[Pos] = goal

    # Walk backward through the parent map until the start is reached.
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)

    # Reverse the backward path so it runs from start to goal.
    path.reverse()

    # Final safety check to make sure the reconstructed path really starts correctly.
    if not path or path[0] != start:
        return []

    return path


# Runs A* from the start cell to the goal cell on the given grid.
def a_star_search(grid: Grid, start: Pos, goal: Pos) -> SearchResult:
    # Priority queue stores (f-score, node), where f = g + h.
    pq: List[Tuple[float, Pos]] = []

    # g_score stores the cheapest known cost from start to each node.
    g_score: Dict[Pos, float] = {start: 0.0}

    # parent stores how each node was reached so the final path can be rebuilt.
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    # The start node begins with cost-so-far 0 and heuristic distance to goal.
    f0 = 0.0 + float(manhattan(start, goal))
    heapq.heappush(pq, (f0, start))

    # Counts how many nodes are actually expanded.
    nodes_expanded = 0

    # Continue until there are no more frontier nodes to explore.
    while pq:
        f, node = heapq.heappop(pq)

        # Ignore stale queue entries if a better route to this node was found later.
        expected_f = g_score.get(node, float("inf")) + float(manhattan(node, goal))
        if f != expected_f:
            continue

        nodes_expanded += 1

        # Stop once the goal is selected for expansion.
        if node == goal:
            break

        current_g = g_score[node]

        # Check each valid neighbour and relax the edge if a cheaper route is found.
        for nbr in grid.neighbors(node):
            step = grid.step_cost(nbr)

            # A* requires non-negative movement costs.
            if step < 0:
                raise ValueError("Negative cost detected (A* requires non-negative costs).")

            new_g = current_g + float(step)

            # Update neighbour data only if this route is better than any previous one.
            if new_g < g_score.get(nbr, float("inf")):
                g_score[nbr] = new_g
                parent[nbr] = node

                # Push the updated neighbour using its new f-score.
                new_f = new_g + float(manhattan(nbr, goal))
                heapq.heappush(pq, (new_f, nbr))

    # Build the final path and read the total cost to the goal.
    path = reconstruct_path(parent, start, goal)
    total = g_score.get(goal, float("inf"))

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        best_cost=g_score,
        parent=parent,
    )