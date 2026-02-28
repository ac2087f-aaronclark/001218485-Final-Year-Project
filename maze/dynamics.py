# maze/dynamics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from maze.grid import Grid, Pos


@dataclass
class SpikeSystem:
    """
    Tracks which cells are currently spiked so we can clear them efficiently.
    """
    spike_cost: int = 50
    k: int = 5
    m: int = 5
    active_spikes: set[Pos] = None

    def __post_init__(self) -> None:
        if self.active_spikes is None:
            self.active_spikes = set()


def clear_spikes(grid: Grid, spikes: SpikeSystem) -> List[Pos]:
    changed: List[Pos] = []
    for (r, c) in spikes.active_spikes:
        if grid.spike_cost[r, c] != 0:
            grid.spike_cost[r, c] = 0
            changed.append((r, c))
    spikes.active_spikes.clear()
    return changed


def window_cells(grid: Grid, center: Pos, k: int) -> List[Pos]:
    """
    Returns all cells in a kxk window centered on center (clipped to bounds).
    k should be odd (5, 7, ...).
    """
    cr, cc = center
    half = k // 2

    out: List[Pos] = []
    for r in range(cr - half, cr + half + 1):
        for c in range(cc - half, cc + half + 1):
            p = (r, c)
            if grid.in_bounds(p) and grid.passable(p):
                out.append(p)
    return out


def apply_spikes_rule3(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    """
    Rule 3:
    - In kxk window around agent
    - Pick m lowest BASE-cost eligible cells
    - Set spike_cost on them (spike persists until next update)
    Eligibility excludes agent cell, start, goal.
    Returns list of changed cells (newly spiked).
    """
    eligible: List[Pos] = []

    for p in window_cells(grid, agent_pos, spikes.k):
        if p == agent_pos:
            continue
        if p == grid.start or p == grid.goal:
            continue
        eligible.append(p)

    # sort by base cost (low -> high)
    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])

    chosen = eligible[: spikes.m]

    changed: List[Pos] = []
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost:
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed


def update_spikes(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    """
    One full update step:
    - clear all previous spikes
    - apply new spikes with Rule 3
    Returns all cells whose spike_cost changed.
    """
    changed = []
    changed.extend(clear_spikes(grid, spikes))
    changed.extend(apply_spikes_rule3(grid, spikes, agent_pos))
    return changed