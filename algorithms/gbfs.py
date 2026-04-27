from __future__ import annotations

# Implements Greedy Best First Search for the weighted 4 connected grid.
# It expands nodes using only Manhattan distance to the goal.

import heapq  #also uses a priority queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from maze.grid import Grid, Pos

# Algorithm code credit goes to:
# https://github.com/shai-raz/AStar-GreedyBestFirstSearch-Pathfinder-with-GUI/blob/master/alogs.py


# Data class stores results of algorithm - integrating it for usage
@dataclass
class SearchResult:
    path: List[Pos]
    total_cost: float
    nodes_expanded: int
    parent: Dict[Pos, Optional[Pos]]


# Returns Manhattan distance between two grid positions.
def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Rebuilds the path from goal back to start using the parent map.
def reconstruct_path(parent: Dict[Pos, Optional[Pos]], start: Pos, goal: Pos) -> List[Pos]:
    # If the goal was never discovered then no path exists.
    if goal not in parent:
        return []

    path: List[Pos] = []
    cur: Optional[Pos] = goal

    # Follow parent links backward until the start is reached.
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)

    # Reverse the backward path so it runs from start to goal.
    path.reverse()

    # Safety check to make sure the reconstructed path begins at the start cell.
    if not path or path[0] != start:
        return []

    return path


# Unique to gbfs calculates the total path cost after path is chosen
def path_cost(grid: Grid, path: List[Pos]) -> float:
    # Cost is charged when entering each cell, so the start cell is excluded.
    if not path:
        return float("inf")

    total = 0.0
    for i in range(1, len(path)):
        total += float(grid.step_cost(path[i])) #loops through and adds total cost

    return total


# Runs Greedy Best First Search from the start cell to the goal cell.
def greedy_best_first_search(grid: Grid, start: Pos, goal: Pos) -> SearchResult:
    # Priority queue stores nodes appearing closest to nodes
    pq: List[Tuple[float, Pos]] = []
    heapq.heappush(pq, (float(manhattan(start, goal)), start))

    # It remembers how each cell was discovered by looking at the parent
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    # This records nodes that have already been expanded.
    visited: set[Pos] = set()

    # Counts how many nodes are expanded during the search.
    nodes_expanded = 0

    while pq:
        _, node = heapq.heappop(pq) #removes node with lowest heurisitc distance to list

        # Skip nodes already expanded earlier.
        if node in visited:
            continue

        visited.add(node)
        nodes_expanded += 1

        # Stop as soon as the goal is reached.
        if node == goal:
            break

        # Expand each valid neighbour using heuristic priority only.
        for nbr in grid.neighbors(node):
            if nbr in visited:
                continue

            # Sticks in one direction that looks the closest
            # GBFS does not revise nodes using a cheaper cost like UCS or A*.
            if nbr not in parent:
                parent[nbr] = node

            heapq.heappush(pq, (float(manhattan(nbr, goal)), nbr))

    # Rebuilds the path from priority queue and then calculate its final weighted cost.
    path = reconstruct_path(parent, start, goal)
    total = path_cost(grid, path)

    return SearchResult(
        path=path,
        total_cost=total,
        nodes_expanded=nodes_expanded,
        parent=parent,
    )