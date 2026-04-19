from __future__ import annotations

# Manages the dynamic spike cost behaviour used in the project.
# It supports Local Cost Spiking and Path Ahead Spiking.

from dataclasses import dataclass
from typing import List

from maze.grid import Grid, Pos


# Stores the spike settings and tracks which cells are currently spiked.
@dataclass
class SpikeSystem:
    spike_cost: int = 50
    k: int = 5
    m: int = 5
    active_spikes: set[Pos] = None

    # Creates the active spike set the first time the object is made.
    def __post_init__(self) -> None:
        if self.active_spikes is None:
            self.active_spikes = set()


# Clears all current spikes from the grid and returns the cells that changed.
def clear_spikes(grid: Grid, spikes: SpikeSystem) -> List[Pos]:
    changed: List[Pos] = []

    # Reset each currently spiked cell back to zero extra spike cost.
    for (r, c) in spikes.active_spikes:
        if grid.spike_cost[r, c] != 0:
            grid.spike_cost[r, c] = 0
            changed.append((r, c))

    # Empty the tracked spike set once they have been cleared.
    spikes.active_spikes.clear()
    return changed


# Returns all passable cells inside a k x k window centred on the chosen cell.
def window_cells(grid: Grid, center: Pos, k: int) -> List[Pos]:
    cr, cc = center
    half = k // 2

    out: List[Pos] = []

    # Scan every cell in the clipped square window around the centre.
    for r in range(cr - half, cr + half + 1):
        for c in range(cc - half, cc + half + 1):
            p = (r, c)
            if grid.in_bounds(p) and grid.passable(p):
                out.append(p)

    return out


# Applies Local Cost Spiking around the agent and returns the newly changed cells.
def apply_spikes_local_cost_spiking(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    eligible: List[Pos] = []

    # Collect valid local cells but exclude the agent, start, and goal cells.
    for p in window_cells(grid, agent_pos, spikes.k):
        if p == agent_pos:
            continue
        if p == grid.start or p == grid.goal:
            continue
        eligible.append(p)

    # Sort by base cost so the lowest cost cells are chosen first.
    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])
    chosen = eligible[: spikes.m]

    changed: List[Pos] = []

    # Apply the spike cost to the chosen cells and track which ones changed.
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost:
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed


# Performs one full Local Cost Spiking update by clearing old spikes and applying new ones.
def update_spikes_local_cost_spiking(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    changed = []
    changed.extend(clear_spikes(grid, spikes))
    changed.extend(apply_spikes_local_cost_spiking(grid, spikes, agent_pos))
    return changed


# Applies Path Ahead Spiking along the current planned path and returns the newly changed cells.
def apply_spikes_path_ahead_spiking(
    grid: Grid,
    spikes: SpikeSystem,
    path: List[Pos],
    agent_pos: Pos,
    lookahead: int = 10,
) -> List[Pos]:
    # No path means there is nothing ahead to spike.
    if not path:
        return []

    # Find where the agent currently sits on the path.
    try:
        idx = path.index(agent_pos)
    except ValueError:
        idx = 0

    # Take the next lookahead cells ahead of the current agent position.
    ahead = path[idx + 1: idx + 1 + lookahead]

    eligible: List[Pos] = []

    # Exclude the agent, start, and goal cells from being spiked.
    for p in ahead:
        if p == agent_pos:
            continue
        if p == grid.start or p == grid.goal:
            continue
        eligible.append(p)

    # Sort by base cost so the lowest cost cells are chosen first.
    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])
    chosen = eligible[: spikes.m]

    changed: List[Pos] = []

    # Apply the spike cost to the chosen ahead cells and track which ones changed.
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost:
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed


# Performs one full Path Ahead Spiking update by clearing old spikes and applying new ones.
def update_spikes_path_ahead_spiking(
    grid: Grid,
    spikes: SpikeSystem,
    agent_pos: Pos,
    path: List[Pos],
    lookahead: int = 10,
) -> List[Pos]:
    changed = []
    changed.extend(clear_spikes(grid, spikes))
    changed.extend(
        apply_spikes_path_ahead_spiking(grid, spikes, path, agent_pos, lookahead=lookahead)
    )
    return changed