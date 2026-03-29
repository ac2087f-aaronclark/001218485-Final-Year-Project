# baseline_experiment.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List
import time
import csv

from maze.grid import Grid, GridSpec, SMALL, MEDIUM, LARGE

from algorithms.ucs import uniform_cost_search
from algorithms.a_star import a_star_search
from algorithms.weighted_a_star import weighted_a_star_search
from algorithms.gbfs import greedy_best_first_search
from algorithms.d_star_lite import DStarLite


@dataclass
class RunResult:
    algorithm: str
    size: str
    seed: int
    found: bool
    path_len: int
    total_cost: float
    nodes_expanded: int
    runtime_ms: float
    extra: Dict[str, Any]


def run_algorithm(grid: Grid, algo: str, w: float = 2.0) -> RunResult:
    t0 = time.perf_counter()

    if algo == "UCS":
        res = uniform_cost_search(grid, grid.start, grid.goal)
        found = bool(res.path)
        path_len = len(res.path)
        total_cost = float(res.total_cost)
        nodes_expanded = int(res.nodes_expanded)
        extra = {}

    elif algo == "A*":
        res = a_star_search(grid, grid.start, grid.goal)
        found = bool(res.path)
        path_len = len(res.path)
        total_cost = float(res.total_cost)
        nodes_expanded = int(res.nodes_expanded)
        extra = {}

    elif algo == "wA*":
        res = weighted_a_star_search(grid, grid.start, grid.goal, w=w)
        found = bool(res.path)
        path_len = len(res.path)
        total_cost = float(res.total_cost)
        nodes_expanded = int(res.nodes_expanded)
        extra = {"w": w}

    elif algo == "GBFS":
        res = greedy_best_first_search(grid, grid.start, grid.goal)
        found = bool(res.path)
        path_len = len(res.path)
        total_cost = float(res.total_cost)
        nodes_expanded = int(res.nodes_expanded)
        extra = {}

    elif algo == "D*Lite":
        dstar = DStarLite(grid, grid.start, grid.goal)
        path = dstar.plan_path()
        found = bool(path)
        path_len = len(path)
        # D* Lite file currently doesn’t return total cost, so compute it here
        total_cost = 0.0
        for i in range(1, len(path)):
            total_cost += float(grid.step_cost(path[i]))
        nodes_expanded = int(dstar.nodes_expanded)
        extra = {}

    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    t1 = time.perf_counter()

    return RunResult(
        algorithm=algo,
        size=f"{grid.rows}x{grid.cols}",
        seed=grid.seed if grid.seed is not None else -1,
        found=found,
        path_len=path_len,
        total_cost=total_cost if found else float("inf"),
        nodes_expanded=nodes_expanded,
        runtime_ms=(t1 - t0) * 1000.0,
        extra=extra,
    )


def run_batch(
    specs: List[GridSpec],
    seeds: List[int],
    algos: List[str],
    w: float = 2.0,
    out_csv: str = "baseline_results.csv",
) -> None:
    rows: List[RunResult] = []

    for spec in specs:
        for seed in seeds:
            grid = Grid(spec, seed=seed)

            for algo in algos:
                result = run_algorithm(grid, algo, w=w)
                rows.append(result)
                print(
                    f"{result.algorithm:7}  {result.size:7}  seed={seed:3}  "
                    f"found={result.found}  cost={result.total_cost:.1f}  "
                    f"len={result.path_len:3}  exp={result.nodes_expanded:5}  "
                    f"{result.runtime_ms:8.2f} ms"
                )

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "size", "seed", "found",
            "path_len", "total_cost", "nodes_expanded", "runtime_ms",
            "w"
        ])
        for r in rows:
            writer.writerow([
                r.algorithm,
                r.size,
                r.seed,
                int(r.found),
                r.path_len,
                r.total_cost,
                r.nodes_expanded,
                r.runtime_ms,
                r.extra.get("w", ""),
            ])

    print(f"\nSaved: {out_csv}")