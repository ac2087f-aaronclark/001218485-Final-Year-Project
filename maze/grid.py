from __future__ import annotations

"""
This file defines the core grid used throughout the project.

It creates an open weighted grid with boundary walls, fixed start and goal
positions, and support for base traversal costs and temporary spike costs.
"""

import random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


Pos = Tuple[int, int]  # (row, col)


# Stores the row and column dimensions for a grid size.
@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int


SMALL = GridSpec(20, 20)
MEDIUM = GridSpec(40, 40)
LARGE = GridSpec(80, 80)


# Represents the weighted grid world used by the algorithms.
class Grid:
    """
    Open grid world:
    - 4-connected movement
    - only boundary walls (outer border blocked)
    - base traversal costs in [1, 9] for traversable cells
    """

    # Builds the grid, boundary walls, fixed start/goal positions, and base costs.
    def __init__(self, spec: GridSpec, seed: Optional[int] = 0):
        self.rows = spec.rows
        self.cols = spec.cols
        self.seed = seed

        # walls[r, c] == True means blocked
        self.walls = np.zeros((self.rows, self.cols), dtype=bool)

        # Boundary walls
        self.walls[0, :] = True
        self.walls[self.rows - 1, :] = True
        self.walls[:, 0] = True
        self.walls[:, self.cols - 1] = True

        # Start/goal: opposite corners inside the boundary
        self.start: Pos = (self.rows - 2, 1)   # bottom-left interior
        self.goal: Pos = (1, self.cols - 2)    # top-right interior

        # Base traversal costs (1..9) for traversable cells
        self.base_cost = np.ones((self.rows, self.cols), dtype=np.int16)

        rng = random.Random(self.seed)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.walls[r, c]:
                    continue
                self.base_cost[r, c] = rng.randint(1, 9)

        # Ensure start/goal are always normal-cost cells
        self.base_cost[self.start] = 1
        self.base_cost[self.goal] = 1

        # Dynamic spike costs (0 normally; spike cells temporarily set to e.g. 50)
        self.spike_cost = np.zeros((self.rows, self.cols), dtype=np.int16)

    # Checks whether a position lies inside the grid bounds.
    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    # Checks whether a position is traversable.
    def passable(self, p: Pos) -> bool:
        r, c = p
        return not self.walls[r, c]

    # Yields valid 4-connected neighbours in the current fixed order.
    def neighbors(self, p: Pos) -> Iterable[Pos]:
        r, c = p
        candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        for nr, nc in candidates:
            if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.walls[nr, nc]:
                yield nr, nc

    # Returns the cost of entering the destination cell.
    def step_cost(self, to_pos: Pos) -> int:
        """Cost to ENTER the destination cell."""
        r, c = to_pos
        return int(self.base_cost[r, c] + self.spike_cost[r, c])