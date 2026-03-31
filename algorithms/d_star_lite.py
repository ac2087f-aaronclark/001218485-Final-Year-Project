from __future__ import annotations

"""
This file implements the D* Lite algorithm for the project.

It supports incremental replanning on the weighted grid after dynamic
cost changes, while reusing previously computed search information.
"""

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from maze.grid import Grid, Pos

# Credit for code goes to:
# https://github.com/Sollimann/Dstar-lite-pathplanner/blob/master/python/python/d_star_lite.py

INF = float("inf")


# Stores a D* Lite priority key as the ordered pair (k1, k2).
@dataclass(order=True, frozen=True)
class Priority:
    k1: float
    k2: float


# Returns the Manhattan-distance heuristic between two grid cells.
def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Min-heap with lazy deletion for D* Lite priority queue operations.
class PriorityQueue:
    """
    Min-heap with lazy deletion.
    Stores (Priority, node). Supports "update" by pushing a new entry and
    remembering the best key per node; stale entries are skipped later.
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[Priority, Pos]] = []
        self._best_key: Dict[Pos, Priority] = {}

    # Returns the smallest current key in the queue.
    def top_key(self) -> Priority:
        self._purge()
        if not self._heap:
            return Priority(INF, INF)
        return self._heap[0][0]

    # Checks whether the queue currently contains the given node.
    def contains(self, node: Pos) -> bool:
        return node in self._best_key

    # Inserts a new node or updates the stored key for an existing one.
    def push_or_update(self, node: Pos, key: Priority) -> None:
        self._best_key[node] = key
        heapq.heappush(self._heap, (key, node))

    # Removes the current key record for a node.
    def remove(self, node: Pos) -> None:
        self._best_key.pop(node, None)

    # Pops and returns the current smallest valid queue entry.
    def pop(self) -> Tuple[Priority, Pos]:
        """
        Pop and return (k_old, u) where k_old is the key that was actually popped.
        This is important for correct D* Lite logic.
        """
        self._purge()
        key, node = heapq.heappop(self._heap)

        # By construction (after purge), this is the current valid entry.
        del self._best_key[node]
        return key, node

    # Removes stale heap entries until the top entry is current.
    def _purge(self) -> None:
        while self._heap:
            key, node = self._heap[0]
            cur = self._best_key.get(node)

            if cur is None:
                heapq.heappop(self._heap)
                continue

            if cur != key:
                heapq.heappop(self._heap)
                continue

            break


# Implements D* Lite over the project's weighted 4-connected grid.
class DStarLite:
    """
    D* Lite adapted to your Grid.

    - State: (row, col)
    - succ/pred: 4-connected neighbors (same in this undirected grid)
    - c(u,v): cost to ENTER v -> grid.step_cost(v)
    """

    def __init__(self, grid: Grid, s_start: Pos, s_goal: Pos):
        self.grid = grid
        self.s_start = s_start
        self.s_goal = s_goal

        self.s_last = s_start
        self.k_m = 0.0

        self.g: Dict[Pos, float] = {}
        self.rhs: Dict[Pos, float] = {}

        self.U = PriorityQueue()

        self.nodes_expanded = 0  # metric

        # Initial setup: only the goal starts with rhs = 0.
        self._set_rhs(self.s_goal, 0.0)
        self.U.push_or_update(self.s_goal, self._calculate_key(self.s_goal))

    # ---------- dict helpers ----------

    # Returns the current g-value for a state.
    def _get_g(self, s: Pos) -> float:
        return self.g.get(s, INF)

    # Sets the g-value for a state.
    def _set_g(self, s: Pos, v: float) -> None:
        self.g[s] = v

    # Returns the current rhs-value for a state.
    def _get_rhs(self, s: Pos) -> float:
        return self.rhs.get(s, INF)

    # Sets the rhs-value for a state.
    def _set_rhs(self, s: Pos, v: float) -> None:
        self.rhs[s] = v

    # ---------- graph primitives ----------

    # Returns the successor states of a node.
    def succ(self, s: Pos) -> Iterable[Pos]:
        return self.grid.neighbors(s)

    # Returns the predecessor states of a node.
    def pred(self, s: Pos) -> Iterable[Pos]:
        # Undirected 4-neighbour grid -> pred = succ
        return self.grid.neighbors(s)

    # Returns the transition cost from u to v.
    def c(self,u: Pos, v: Pos) -> float:
        if not self.grid.passable(v):
            return INF
        return float(self.grid.step_cost(v))

    # ---------- core algorithm ----------

    # Calculates the current priority key for a state.
    def _calculate_key(self, s: Pos) -> Priority:
        g_rhs = min(self._get_g(s), self._get_rhs(s))
        return Priority(
            g_rhs + float(manhattan(self.s_start, s)) + self.k_m,
            g_rhs,
        )

    # Updates one vertex according to the D* Lite consistency rules.
    def _update_vertex(self, u: Pos) -> None:
        if u != self.s_goal:
            best = INF
            for s2 in self.succ(u):
                best = min(best, self.c(u, s2) + self._get_g(s2))
            self._set_rhs(u, best)

        if self.U.contains(u):
            self.U.remove(u)

        if self._get_g(u) != self._get_rhs(u):
            self.U.push_or_update(u, self._calculate_key(u))

    # Repeatedly processes inconsistent states until the current start is locally consistent.
    def compute_shortest_path(self) -> None:
        while (self.U.top_key() < self._calculate_key(self.s_start)) or (
            self._get_rhs(self.s_start) != self._get_g(self.s_start)
        ):
            k_old, u = self.U.pop()
            self.nodes_expanded += 1

            k_new = self._calculate_key(u)

            if k_old < k_new:
                # Key became worse -> reinsert with updated key.
                self.U.push_or_update(u, k_new)

            elif self._get_g(u) > self._get_rhs(u):
                # Overconsistent -> set g to rhs, then update predecessors.
                self._set_g(u, self._get_rhs(u))
                for p in self.pred(u):
                    self._update_vertex(p)

            else:
                # Underconsistent -> reset g, then update u and predecessors.
                self._set_g(u, INF)
                self._update_vertex(u)
                for p in self.pred(u):
                    self._update_vertex(p)

    # Selects the next path state that minimises one-step cost plus successor g-value.
    def _argmin_next(self, s: Pos) -> Optional[Pos]:
        best = INF
        best_s: Optional[Pos] = None

        for s2 in self.succ(s):
            val = self.c(s, s2) + self._get_g(s2)
            if val < best:
                best = val
                best_s = s2

        return best_s

    # ---------- public API ----------

    # Updates the current start state for replanning.
    def set_start(self, new_start: Pos) -> None:
        self.s_start = new_start

    # Plans a path from the current start state to the goal state.
    def plan_path(self) -> List[Pos]:
        """
        Plan a path from current s_start to s_goal.
        """
        self.compute_shortest_path()

        if self._get_rhs(self.s_start) == INF:
            return []

        path = [self.s_start]
        cur = self.s_start
        max_steps = self.grid.rows * self.grid.cols  # safety

        for _ in range(max_steps):
            if cur == self.s_goal:
                break

            nxt = self._argmin_next(cur)
            if nxt is None:
                return []
            if self.c(cur, nxt) == INF:
                return []

            cur = nxt
            path.append(cur)

        if path and path[-1] == self.s_goal:
            return path
        return []

    # Notifies D* Lite that entry costs changed for the given cells.
    def notify_cost_changes(self, changed_cells: List[Pos]) -> None:
        """
        Call after costs changed (spikes later).
        'changed_cells' should be cells whose entry-cost changed.
        """
        self.k_m += float(manhattan(self.s_last, self.s_start))
        self.s_last = self.s_start

        affected = set()
        for v in changed_cells:
            affected.add(v)
            for p in self.pred(v):
                affected.add(p)

        for u in affected:
            self._update_vertex(u)