from __future__ import annotations

# Implements D* Lite for the weighted grid.
# It supports incremental replanning after dynamic cost changes.

import heapq #D* Lite uses a priority queue with two parts and a Manhattan heuristic,
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from maze.grid import Grid, Pos

# Credit for code goes to:
# https://github.com/Sollimann/Dstar-lite-pathplanner/blob/master/python/python/d_star_lite.py

INF = float("inf")


# Stores 2 D* Lite priority keys for the priority queue (k1 k2).
@dataclass(order=True, frozen=True)
class Priority:
    k1: float
    k2: float


# Returns Manhattan distance between two grid positions.
def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Priority queue with lazy deletion structure technique for D* Lite.
# It stores (Priority node) pairs and skips stale entries later.
# It stores nodes that need to be processed or repaired.
class PriorityQueue:
    def __init__(self) -> None:
        self._heap: List[Tuple[Priority, Pos]] = []
        self._best_key: Dict[Pos, Priority] = {}

    # Returns the smallest valid key currently in the queue.
    def top_key(self) -> Priority:
        self._purge()
        if not self._heap:
            return Priority(INF, INF)
        return self._heap[0][0]

    # Checks whether a node currently has a valid entry in the queue.
    def contains(self, node: Pos) -> bool:
        return node in self._best_key

    # Inserts a node or updates its best known key.
    def push_or_update(self, node: Pos, key: Priority) -> None:
        self._best_key[node] = key
        heapq.heappush(self._heap, (key, node))

    # Removes the current valid key record for a node.
    def remove(self, node: Pos) -> None:
        self._best_key.pop(node, None)

    # Pops and returns the smallest valid queue entry.
    def pop(self) -> Tuple[Priority, Pos]:
        self._purge()
        key, node = heapq.heappop(self._heap)

        # After purging this must be the current valid entry for the node.
        del self._best_key[node]
        return key, node

    # Removes stale heap entries until the top entry matches the current best key.
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

# Unlike the others as functions, this is stored as a class - access memory across time
# D* Lite adapted to the projects weighted 4 connected grid.
class DStarLite:
    # States are grid positions.
    # Successors and predecessors are the same in this undirected grid.
    # c(u v) is the cost to enter v.
#this creates the planner for D* lite
    def __init__(self, grid: Grid, s_start: Pos, s_goal: Pos):
        self.grid = grid
        self.s_start = s_start
        self.s_goal = s_goal

        # s_last is used when updating the km offset after movement.
        self.s_last = s_start
        self.k_m = 0.0

        # g stores current best path estimates and rhs stores one step lookahead values.
        self.g: Dict[Pos, float] = {}
        self.rhs: Dict[Pos, float] = {}

        self.U = PriorityQueue()

        # Tracks how many queue states were processed.
        self.nodes_expanded = 0

        # Initial setup starts with rhs(goal) = 0 and pushes the goal into the queue.
        self._set_rhs(self.s_goal, 0.0)
        self.U.push_or_update(self.s_goal, self._calculate_key(self.s_goal))

    # Returns the current g value for a state.
    def _get_g(self, s: Pos) -> float:
        return self.g.get(s, INF)

    # Sets the g value for a state.
    def _set_g(self, s: Pos, v: float) -> None:
        self.g[s] = v

    # Returns the current rhs value for a state.
    def _get_rhs(self, s: Pos) -> float:
        return self.rhs.get(s, INF)

    # Sets the rhs value for a state.
    def _set_rhs(self, s: Pos, v: float) -> None:
        self.rhs[s] = v

    # Returns the successor states of a node.
    def succ(self, s: Pos) -> Iterable[Pos]:
        return self.grid.neighbors(s)

    # Returns the predecessor states of a node.
    def pred(self, s: Pos) -> Iterable[Pos]:
        # In this undirected 4 connected grid predecessors equal successors.
        return self.grid.neighbors(s)

    # Returns the transition cost from u to v.
    def c(self, u: Pos, v: Pos) -> float:
        if not self.grid.passable(v):
            return INF
        return float(self.grid.step_cost(v))

    # Calculates the current priority key for a state.
    def _calculate_key(self, s: Pos) -> Priority:
        g_rhs = min(self._get_g(s), self._get_rhs(s))
        return Priority(
            g_rhs + float(manhattan(self.s_start, s)) + self.k_m,
            g_rhs,
        )

    # Updates one vertex using the D* Lite consistency rules.
    def _update_vertex(self, u: Pos) -> None:
        if u != self.s_goal:
            best = INF
            for s2 in self.succ(u):
                best = min(best, self.c(u, s2) + self._get_g(s2))
            self._set_rhs(u, best)

        if self.U.contains(u):
            self.U.remove(u)

        # Only inconsistent states stay in the queue.
        if self._get_g(u) != self._get_rhs(u):
            self.U.push_or_update(u, self._calculate_key(u))

    # Runs the search
    def compute_shortest_path(self) -> None:
        while (self.U.top_key() < self._calculate_key(self.s_start)) or (
            self._get_rhs(self.s_start) != self._get_g(self.s_start)
        ):
            k_old, u = self.U.pop()
            self.nodes_expanded += 1

            k_new = self._calculate_key(u)

            # If the key became worse reinsert the node with its updated key.
            if k_old < k_new:
                self.U.push_or_update(u, k_new)

            # If overconsistent set g to rhs and update predecessors.
            elif self._get_g(u) > self._get_rhs(u):
                self._set_g(u, self._get_rhs(u))
                for p in self.pred(u):
                    self._update_vertex(p)

            # If underconsistent reset g then update u and its predecessors.
            else:
                self._set_g(u, INF)
                self._update_vertex(u)
                for p in self.pred(u):
                    self._update_vertex(p)

    # Chooses the next path state with the smallest one step cost plus successor g value.
    def _argmin_next(self, s: Pos) -> Optional[Pos]:
        best = INF
        best_s: Optional[Pos] = None

        for s2 in self.succ(s):
            val = self.c(s, s2) + self._get_g(s2)
            if val < best:
                best = val
                best_s = s2

        return best_s

    # Updates the current start state before replanning.
    def set_start(self, new_start: Pos) -> None:
        self.s_start = new_start

    # Plans a path from the current start state to the goal.
    def plan_path(self) -> List[Pos]:
        self.compute_shortest_path()

        # If rhs(start) is infinite no route to the goal is known.
        if self._get_rhs(self.s_start) == INF:
            return []

        path = [self.s_start]
        cur = self.s_start

        # Safety limit prevents infinite loops if something goes wrong.
        max_steps = self.grid.rows * self.grid.cols

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

    # Tells D* Lite that entry costs changed for the given cells.
    def notify_cost_changes(self, changed_cells: List[Pos]) -> None:
        # km shifts the heuristic reference after the agent has moved.
        self.k_m += float(manhattan(self.s_last, self.s_start))
        self.s_last = self.s_start

        # A changed cell and its predecessors may now be inconsistent.
        affected = set()
        for v in changed_cells:
            affected.add(v)
            for p in self.pred(v):
                affected.add(p)

        for u in affected:
            self._update_vertex(u)