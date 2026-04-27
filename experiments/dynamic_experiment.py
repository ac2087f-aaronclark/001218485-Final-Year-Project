from __future__ import annotations

# Runs one dynamic pathfinding episode for the project.
# It applies periodic cost spikes, replans when needed, and records the main metrics.

# for baseline create grid → run algorithm once → save result
# for dynamic create grid → plan path → move agent step by step
# → every 10 steps apply spikes → replan → continue until goal
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from algorithms.a_star import a_star_search
from algorithms.d_star_lite import DStarLite
from algorithms.gbfs import greedy_best_first_search
from algorithms.ucs import uniform_cost_search
from algorithms.weighted_a_star import weighted_a_star_search
from maze.dynamic_costs import (
    SpikeSystem,
    update_spikes_local_cost_spiking,
    update_spikes_path_ahead_spiking,
)
from maze.grid import Grid, Pos


# Stores the result of one dynamic run.
@dataclass
class DynamicRunResult:
    algorithm: str
    size: str
    seed: int
    found: bool

    steps_taken: int
    total_cost: float

    replans: int
    updates: int

    total_nodes_expanded: int
    total_replan_time_ms: float
    total_runtime_ms: float

    extra: Dict[str, Any]


# Runs one fresh planning call for incremental algorithms, same code as baseline
def plan_once(grid: Grid, algo: str, start: Pos, goal: Pos, w: float = 2.0):
    # Measure only the planning call inside this helper.
    t0 = time.perf_counter()

    if algo == "UCS":
        res = uniform_cost_search(grid, start, goal)
        path = res.path
        expanded = int(res.nodes_expanded)
        extra = {}

    elif algo == "A*":
        res = a_star_search(grid, start, goal)
        path = res.path
        expanded = int(res.nodes_expanded)
        extra = {}

    elif algo == "wA*":
        res = weighted_a_star_search(grid, start, goal, w=w)
        path = res.path
        expanded = int(res.nodes_expanded)
        extra = {"w": w}

    elif algo == "GBFS":
        res = greedy_best_first_search(grid, start, goal)
        path = res.path
        expanded = int(res.nodes_expanded)
        extra = {}

    else:
        raise ValueError(f"plan_once does not support: {algo}")

    t1 = time.perf_counter()

    # Returns the path, node expansions, planning time, and any extra settings used.
    return path, expanded, (t1 - t0) * 1000.0, extra


# Runs one full dynamic episode including movement, updates, and replanning.
#one algorithm + one grid + one seed + one dynamic rule
# inputs settings
def run_dynamic_episode(
    grid: Grid,
    algo: str,
    *,
    seed: int,
    n: int = 10,
    k: int = 5,
    m: int = 5,
    spike_cost: int = 50,
    w: float = 2.0,
    rule: str = "Local Cost Spiking",
    lookahead: int = 10,
    step_limit: Optional[int] = None, #for safety limit
) -> DynamicRunResult:
    # Starts total episode timing for (planning + movement loop + spike updates + replanning)
    episode_t0 = time.perf_counter()

    # Default safety limit stops runs from going on forever. step limit
    if step_limit is None:
        step_limit = grid.rows * grid.cols * 20

    # Creates the spike controller with the chosen disruption settings.
    spikes = SpikeSystem(k=k, m=m, spike_cost=spike_cost)

    agent = grid.start
    goal = grid.goal

    steps_taken = 0
    total_cost = 0.0

    replans = 0
    updates = 0

    total_nodes_expanded = 0
    total_replan_time_ms = 0.0

    # Stores the persistent D* Lite object when the incremental algorithm is used.
    dstar = None

    # Create the initial path before the agent starts moving.
    if algo == "D*Lite":
        dstar = DStarLite(grid, agent, goal)

        # Count only the additional D* Lite expansions produced by this planning call.
        dstar_prev_expanded = dstar.nodes_expanded
        t0 = time.perf_counter()
        path = dstar.plan_path()
        t1 = time.perf_counter()
        delta_expanded = dstar.nodes_expanded - dstar_prev_expanded

        replans += 1
        total_replan_time_ms += (t1 - t0) * 1000.0
        total_nodes_expanded += int(delta_expanded)

    else:
        # the other algorithms just plan from scratch.
        path, expanded, ms, _ = plan_once(grid, algo, agent, goal, w=w)
        replans += 1
        total_nodes_expanded += expanded
        total_replan_time_ms += ms

    # Stop immediately if no initial path can be found.
    if not path:
        episode_t1 = time.perf_counter()
        return DynamicRunResult(
            algorithm=algo,
            size=f"{grid.rows}x{grid.cols}",
            seed=seed,
            found=False,
            steps_taken=0,
            total_cost=float("inf"),
            replans=replans,
            updates=updates,
            total_nodes_expanded=total_nodes_expanded,
            total_replan_time_ms=total_replan_time_ms,
            total_runtime_ms=(episode_t1 - episode_t0) * 1000.0,
            extra={"reason": "no initial path", "rule": rule},
        )

    # idx points to the agents current position within the planned path once 10 interval happens.
    idx = 0

    # Actual Simulation - Continue until the goal is reached or the step limit is hit.
    while agent != goal and steps_taken < step_limit:
        # If the current path has been fully consumed, plan again from the current agent position.
        if idx >= len(path) - 1:
            if algo == "D*Lite":
                dstar.set_start(agent)

                # Count only the new D* Lite expansions from this replan.
                dstar_prev_expanded = dstar.nodes_expanded
                t0 = time.perf_counter()
                path = dstar.plan_path()
                t1 = time.perf_counter()
                delta_expanded = dstar.nodes_expanded - dstar_prev_expanded

                replans += 1
                total_replan_time_ms += (t1 - t0) * 1000.0
                total_nodes_expanded += int(delta_expanded)
            else:
                path, expanded, ms, _ = plan_once(grid, algo, agent, goal, w=w)
                replans += 1
                total_nodes_expanded += expanded
                total_replan_time_ms += ms

            idx = 0
            if not path:
                break

        # Move one step forward along the currently planned route.
        agent = path[idx + 1]
        idx += 1

        steps_taken += 1

        # Cost is charged when entering the new cell.
        total_cost += float(grid.step_cost(agent))

        # Every N steps, apply a disruption update and then replan.
        if steps_taken % n == 0 and agent != goal:
            updates += 1

            if rule == "Local Cost Spiking":   #LCS happens first
                changed = update_spikes_local_cost_spiking(grid, spikes, agent)
            elif rule == "Path Ahead Spiking":  #then PAS - they are sepeate
                changed = update_spikes_path_ahead_spiking(
                    grid,
                    spikes,
                    agent,
                    path,
                    lookahead=lookahead,
                )
            else:
                raise ValueError(f"Unknown rule: {rule}")

            if algo == "D*Lite":
                # D* Lite keeps its earlier search state and repairs it after changes.
                dstar.set_start(agent)
                dstar.notify_cost_changes(changed)

                # Count only the additional expansions caused by this repair step.
                dstar_prev_expanded = dstar.nodes_expanded
                t0 = time.perf_counter()
                path = dstar.plan_path()
                t1 = time.perf_counter()
                delta_expanded = dstar.nodes_expanded - dstar_prev_expanded

                replans += 1
                total_replan_time_ms += (t1 - t0) * 1000.0
                total_nodes_expanded += int(delta_expanded)
                idx = 0
            else:
                # The other algorithms discard the old path and search again from scratch.
                path, expanded, ms, _ = plan_once(grid, algo, agent, goal, w=w)
                replans += 1
                total_nodes_expanded += expanded
                total_replan_time_ms += ms
                idx = 0 # where the agent currently is inside the planned path list restarts

            if not path:
                break

    # A run is successful only if the agent physically reaches the goal.
    found = (agent == goal)

    # Stores the run settings so they can be written out with the results.
    extra: Dict[str, Any] = {} #for future settings
    if algo == "wA*":
        extra["w"] = w

    extra["rule"] = rule
    if rule == "Path Ahead Spiking":
        extra["lookahead"] = lookahead

    extra["N"] = n
    extra["k"] = k
    extra["m"] = m
    extra["spike_cost"] = spike_cost
    extra["step_limit"] = step_limit

    episode_t1 = time.perf_counter()

    return DynamicRunResult(  #returns all metrics from the run into object
        algorithm=algo,
        size=f"{grid.rows}x{grid.cols}",
        seed=seed,
        found=found,
        steps_taken=steps_taken,
        total_cost=total_cost if found else float("inf"),
        replans=replans,
        updates=updates,
        total_nodes_expanded=total_nodes_expanded,
        total_replan_time_ms=total_replan_time_ms,
        total_runtime_ms=(episode_t1 - episode_t0) * 1000.0,
        extra=extra,
    )