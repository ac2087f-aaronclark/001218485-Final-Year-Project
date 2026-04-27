from __future__ import annotations

# Manages the dynamic spike cost behaviour used in the project.
# It supports Local Cost Spiking and Path Ahead Spiking.

from dataclasses import dataclass
from typing import List

from maze.grid import Grid, Pos


# Stores the input spike settings and tracks which cells are currently spiked.
@dataclass
class SpikeSystem:
    spike_cost: int = 50 # adds 50
    k: int = 5 #LCS window size
    m: int = 5 #numbwe of cells affected
    active_spikes: set[Pos] = None #stores spiked nodes for later

    # Creates the active spike set the first time the dataclass is made
    def __post_init__(self) -> None:
        if self.active_spikes is None:
            self.active_spikes = set()


# Clears all current spikes from the grid and returns the original cells that changed.
def clear_spikes(grid: Grid, spikes: SpikeSystem) -> List[Pos]:
    changed: List[Pos] = [] #records list of cells changed (for D•)

    # Reset each currently spiked cell back to zero extra spike cost.
    for (r, c) in spikes.active_spikes:
        if grid.spike_cost[r, c] != 0:
            grid.spike_cost[r, c] = 0
            changed.append((r, c))

    # Empty the tracked spike set once they have been cleared.
    spikes.active_spikes.clear()
    return changed


# Returns all passable nodes inside a k x k window centred on the agents current location
def window_cells(grid: Grid, center: Pos, k: int) -> List[Pos]:
    cr, cc = center
    half = k // 2
#2 cells in rach direction
    out: List[Pos] = []
#stores valid cells
    # Scan every cell in the clipped square window around the agent within bounries.
    for r in range(cr - half, cr + half + 1):
        for c in range(cc - half, cc + half + 1):
            p = (r, c)
            if grid.in_bounds(p) and grid.passable(p):
                out.append(p)

    return out


# Applies Local Cost Spiking around the agent and returns the newly changed cells.
def apply_spikes_local_cost_spiking(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    eligible: List[Pos] = [] #stores cells to be spiked

    # Collect valid local cells but exclude the agent, start, and goal cells.
    for p in window_cells(grid, agent_pos, spikes.k):
        if p == agent_pos:
            continue
        if p == grid.start or p == grid.goal: #exludes start/end/agent position
            continue
        eligible.append(p)

    # Sort by nodes in the eligable list and chooses the lowest cost cells first.
    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])
    chosen = eligible[: spikes.m] #the first m cells

    changed: List[Pos] = [] #stores which nodes that will change

    # Apply the spike cost to the chosen cells
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost: #ends up adding cell cost + 50
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed   #retunrs spiked nodes


# per update - clears old spikes and repeats process for new location
def update_spikes_local_cost_spiking(grid: Grid, spikes: SpikeSystem, agent_pos: Pos) -> List[Pos]:
    changed = [] #clears spikes
    changed.extend(clear_spikes(grid, spikes))
    changed.extend(apply_spikes_local_cost_spiking(grid, spikes, agent_pos))
    return changed #this is the function that is called for a LCS update


# Applies Path Ahead Spiking.
def apply_spikes_path_ahead_spiking(
    grid: Grid,
    spikes: SpikeSystem,
    path: List[Pos],
    agent_pos: Pos,
    lookahead: int = 10, #amount of nodes chosen in path
) -> List[Pos]:
    # No path means there is nothing ahead to spike.
    if not path:
        return []

    # Find where the agent currently sits on the grid
    try:
        idx = path.index(agent_pos)
    except ValueError:
        idx = 0

    # Take the next lookahead cells ahead of the current agent position.
    ahead = path[idx + 1: idx + 1 + lookahead]

    eligible: List[Pos] = [] #stores the eligable cells

    # Exclude the agent, start, and goal cells from being spiked.
    for p in ahead:
        if p == agent_pos:
            continue
        if p == grid.start or p == grid.goal:
            continue
        eligible.append(p)

    #Just like Sort by base cost so the lowest cost cells are chosen first.
    eligible.sort(key=lambda p: grid.base_cost[p[0], p[1]])
    chosen = eligible[: spikes.m]

    changed: List[Pos] = [] #stores nodes to be changed

    # Apply the spike cost to the chosen ahead cells and track which ones changed.
    for (r, c) in chosen:
        if grid.spike_cost[r, c] != spikes.spike_cost:
            grid.spike_cost[r, c] = spikes.spike_cost
            changed.append((r, c))
        spikes.active_spikes.add((r, c))

    return changed #returns cells


# per update - clears old spikes and repeats process for new location
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