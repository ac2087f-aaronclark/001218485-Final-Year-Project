from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import numpy as np
import random

Pos = Tuple[int, int]  # (row, col)

@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int

SMALL = GridSpec(20, 20)
MEDIUM = GridSpec(40, 40)
LARGE = GridSpec(80, 80)


class Grid:
    """
    Open grid world:
    - 4-connected movement
    - only boundary walls (outer border blocked)
    - base traversal costs in [1, 9] for traversable cells
    """
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

        # Start/goal: opposite corners INSIDE the boundary
        self.start: Pos = (self.rows - 2, 1)  # bottom-left interior
        self.goal: Pos = (1, self.cols - 2)   # top-right interior

        # Base traversal costs (1..9) for traversable cells
        self.base_cost = np.ones((self.rows, self.cols), dtype=np.int16)

        rng = random.Random(self.seed)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.walls[r, c]:
                    continue
                self.base_cost[r, c] = rng.randint(1, 9)

        # (Optional) Ensure start/goal are always normal-cost cells
        self.base_cost[self.start] = 1
        self.base_cost[self.goal] = 1
        # Dynamic spike costs (0 normally; spike cells temporarily set to e.g. 50)
        self.spike_cost = np.zeros((self.rows, self.cols), dtype=np.int16)

    def in_bounds(self, p: Pos) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, p: Pos) -> bool:
        r, c = p
        return not self.walls[r, c]

    def neighbors(self, p: Pos) -> Iterable[Pos]:
        r, c = p
        candidates = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        for nr, nc in candidates:
            if 0 <= nr < self.rows and 0 <= nc < self.cols and not self.walls[nr, nc]:
                yield (nr, nc)

    def step_cost(self, to_pos: Pos) -> int:
        """Cost to ENTER the destination cell."""
        r, c = to_pos
        return int(self.base_cost[r, c] + self.spike_cost[r, c])