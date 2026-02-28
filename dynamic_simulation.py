from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time

from maze.grid import Grid, Pos
from maze.dynamics import SpikeSystem, update_spikes

from algorithms.UCS import uniform_cost_search
from algorithms.A_hash import a_star_search
from algorithms.Weighted_A_hash import weighted_a_star_search
from algorithms.GBFS import greedy_best_first_search
from algorithms.D_hash_lite import DStarLite


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
    total_runtime_ms: float  # NEW: full episode runtime

    extra: Dict[str, Any]


def plan_once(grid: Grid, algo: str, start: Pos, goal: Pos, w: float = 2.0):
    """
    Returns: (path, nodes_expanded, runtime_ms, extra_dict)
    Path includes start and goal.
    """
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
    return path, expanded, (t1 - t0) * 1000.0, extra


def run_dynamic_episode(
    grid: Grid,
    algo: str,
    *,
    seed: int,
    N: int = 10,
    k: int = 5,
    m: int = 5,
    spike_cost: int = 50,
    w: float = 2.0,
    step_limit: Optional[int] = None,
) -> DynamicRunResult:
    """
    Dynamic environment episode:
    - initial plan at start
    - follow path step-by-step
    - every N moves: clear+apply spikes in kxk window around agent, then replan
    """
    episode_t0 = time.perf_counter()  # NEW: start full timer

    if step_limit is None:
        step_limit = grid.rows * grid.cols * 20

    spikes = SpikeSystem(k=k, m=m, spike_cost=spike_cost)

    agent = grid.start
    goal = grid.goal

    steps_taken = 0
    total_cost = 0.0

    replans = 0
    updates = 0

    total_nodes_expanded = 0
    total_replan_time_ms = 0.0

    # ---- D* Lite special handling ----
    dstar = None
    dstar_prev_expanded = 0  # NEW: for delta counting

    if algo == "D*Lite":
        dstar = DStarLite(grid, agent, goal)

        dstar_prev_expanded = dstar.nodes_expanded
        t0 = time.perf_counter()
        path = dstar.plan_path()
        t1 = time.perf_counter()
        delta_expanded = dstar.nodes_expanded - dstar_prev_expanded
        dstar_prev_expanded = dstar.nodes_expanded

        replans += 1
        total_replan_time_ms += (t1 - t0) * 1000.0
        total_nodes_expanded += int(delta_expanded)

    else:
        path, expanded, ms, _ = plan_once(grid, algo, agent, goal, w=w)
        replans += 1
        total_nodes_expanded += expanded
        total_replan_time_ms += ms

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
            total_runtime_ms=(episode_t1 - episode_t0) * 1000.0,  # NEW
            extra={"reason": "no initial path"},
        )

    idx = 0  # path index

    while agent != goal and steps_taken < step_limit:
        # if we ran out of planned steps, replan
        if idx >= len(path) - 1:
            if algo == "D*Lite":
                dstar.set_start(agent)

                dstar_prev_expanded = dstar.nodes_expanded
                t0 = time.perf_counter()
                path = dstar.plan_path()
                t1 = time.perf_counter()
                delta_expanded = dstar.nodes_expanded - dstar_prev_expanded
                dstar_prev_expanded = dstar.nodes_expanded

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

        # move one step along path
        agent = path[idx + 1]
        idx += 1

        steps_taken += 1
        total_cost += float(grid.step_cost(agent))

        # update environment every N moves
        if steps_taken % N == 0 and agent != goal:
            updates += 1
            changed = update_spikes(grid, spikes, agent)

            if algo == "D*Lite":
                dstar.set_start(agent)
                dstar.notify_cost_changes(changed)

                dstar_prev_expanded = dstar.nodes_expanded
                t0 = time.perf_counter()
                path = dstar.plan_path()
                t1 = time.perf_counter()
                delta_expanded = dstar.nodes_expanded - dstar_prev_expanded
                dstar_prev_expanded = dstar.nodes_expanded

                replans += 1
                total_replan_time_ms += (t1 - t0) * 1000.0
                total_nodes_expanded += int(delta_expanded)
                idx = 0

            else:
                path, expanded, ms, _ = plan_once(grid, algo, agent, goal, w=w)
                replans += 1
                total_nodes_expanded += expanded
                total_replan_time_ms += ms
                idx = 0

            if not path:
                break

    found = (agent == goal)

    extra: Dict[str, Any] = {}
    if algo == "wA*":
        extra["w"] = w
    extra["N"] = N
    extra["k"] = k
    extra["m"] = m
    extra["spike_cost"] = spike_cost
    extra["step_limit"] = step_limit

    episode_t1 = time.perf_counter()  # NEW: end full timer

    return DynamicRunResult(
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
        total_runtime_ms=(episode_t1 - episode_t0) * 1000.0,  # NEW
        extra=extra,
    )