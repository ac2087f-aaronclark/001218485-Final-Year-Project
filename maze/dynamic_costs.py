# maze/dynamic_costs.py
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


def apply_spikes_local_cost_spiking(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    """
    Local Cost Spiking:
    - In kxk window around the agent
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

    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])
    chosen = eligible[: spikes.m]

    changed: List[Pos] = []
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost:
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed


def update_spikes_local_cost_spiking(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    """
    One full update step (Local Cost Spiking):
    - clear all previous spikes
    - apply new spikes in the local window around the agent
    Returns all cells whose spike_cost changed.
    """
    changed = []
    changed.extend(clear_spikes(grid, spikes))
    changed.extend(apply_spikes_local_cost_spiking(grid, spikes, agent_pos))
    return changed


# =========================
# Path Ahead Spiking
# =========================

def apply_spikes_path_ahead_spiking(
    grid: Grid,
    spikes: SpikeSystem,
    path: List[Pos],
    agent_pos: Pos,
    lookahead: int = 10,
) -> List[Pos]:
    """
    Path Ahead Spiking:
    - Uses the CURRENT planned path
    - Consider the next `lookahead` cells ahead on the path
    - Select the m lowest BASE-cost eligible cells among those and spike them

    Eligibility excludes agent cell, start, goal.
    Returns list of cells whose spike_cost changed (new spikes applied).
    """
    if not path:
        return []

    # Robustly locate agent on the path (should usually be at index 0)
    try:
        idx = path.index(agent_pos)
    except ValueError:
        idx = 0

    ahead = path[idx + 1: idx + 1 + lookahead]

    eligible: List[Pos] = []
    for p in ahead:
        if p == agent_pos:
            continue
        if p == grid.start or p == grid.goal:
            continue
        eligible.append(p)

    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])
    chosen = eligible[: spikes.m]

    changed: List[Pos] = []
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost:
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed


def update_spikes_path_ahead_spiking(
    grid: Grid,
    spikes: SpikeSystem,
    agent_pos: Pos,
    path: List[Pos],
    lookahead: int = 10,
) -> List[Pos]:
    """
    One full update step (Path Ahead Spiking):
    - clear all previous spikes
    - apply new spikes along the next `lookahead` steps of the current path
    Returns all cells whose spike_cost changed.
    """
    changed = []
    changed.extend(clear_spikes(grid, spikes))
    changed.extend(
        apply_spikes_path_ahead_spiking(grid, spikes, path, agent_pos, lookahead=lookahead)
    )
    return changed