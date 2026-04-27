from __future__ import annotations

# Defines the core weighted grid used throughout the project.
# The grid has boundary walls, fixed start and goal cells, base costs, and temporary spike costs.

import random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


# Grid (node) position stored as row and column.The basic unit
Pos = Tuple[int, int]


# Stores the row and column dimensions for the 3 grid sizes.
@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int


# Fixed grid sizes used in the experiments.
SMALL = GridSpec(20, 20)
MEDIUM = GridSpec(40, 40)
LARGE = GridSpec(80, 80)


# Represents the blueprint for  weighted grid world used by the algorithms.
class Grid:
    # The grid is open inside, uses 4 connected movement, and blocks only the outer border.

    # Creates the grid size, and the seed
    def __init__(self, spec: GridSpec, seed: Optional[int] = 0):
        self.rows = spec.rows #defining detials like row/column amount and seed
        self.cols = spec.cols
        self.seed = seed

        # walls[r, c] is True when the cell is blocked - all open cells are false
        self.walls = np.zeros((self.rows, self.cols), dtype=bool)

        # Block the outer border so the interior stays bounded.
        self.walls[0, :] = True
        self.walls[self.rows - 1, :] = True
        self.walls[:, 0] = True
        self.walls[:, self.cols - 1] = True

        # Fixed start and goal positions are placed in opposite interior corners.
        self.start: Pos = (self.rows - 2, 1)
        self.goal: Pos = (1, self.cols - 2)

        # Base traversal costs start as 1 everywhere and are then randomised for traversable cells.
        self.base_cost = np.ones((self.rows, self.cols), dtype=np.int16)

        # Use a seeded random generator so the same seed always recreates the same weighted map.
        rng = random.Random(self.seed) #same seed equals one map
        for r in range(self.rows):
            for c in range(self.cols):
                if self.walls[r, c]:
                    continue
                self.base_cost[r, c] = rng.randint(1, 9) #for every wall cost give a value from one to 9

        # Keep the start and goal as normal low cost cells.
        self.base_cost[self.start] = 1
        self.base_cost[self.goal] = 1

        # Extra spike costs are stored separately and added on top of the base cost when active to be temporary.
        self.spike_cost = np.zeros((self.rows, self.cols), dtype=np.int16)

#The following are helper functions that algorithms use
    # Checks whether a position lies inside the grid bounds.
    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    # Is this a wall?
    def passable(self, p: Pos) -> bool:
        r, c = p
        return not self.walls[r, c]

    # Limits movement to 4 connected neighbours in the fixed movement order.
    def neighbors(self, p: Pos) -> Iterable[Pos]:
        r, c = p
        candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        for nr, nc in candidates:
            if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.walls[nr, nc]:
                yield nr, nc

    # Returns the cost of entering each cell from another
    def step_cost(self, to_pos: Pos) -> int:
        r, c = to_pos
        return int(self.base_cost[r, c] + self.spike_cost[r, c]) #adding spiking cost 50 to node