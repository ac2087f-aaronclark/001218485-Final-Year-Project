"""
This file runs the dynamic experiment batches for the project
and then creates summary tables from the generated CSV results.
"""

import csv

import pandas as pd

from maze.grid import Grid, SMALL, MEDIUM, LARGE
from experiments.dynamic_experiment import run_dynamic_episode


# Runs the dynamic experiment across multiple sizes, seeds, and algorithms,
# then saves the collected results to a CSV file.
def run_dynamic_batch(
    grid_specs,
    seed_values,
    algorithms,
    *,
    out_csv="dynamic_results.csv",
    w=2.0,
    update_every_n=10,
    k=5,
    m=5,
    spike_cost=50,
    rule="Local Cost Spiking",
    lookahead=10,
):
    rows = []

    for spec in grid_specs:
        for seed in seed_values:
            grid = Grid(spec, seed=seed)

            for algo in algorithms:
                res = run_dynamic_episode(
                    grid,
                    algo,
                    seed=seed,
                    w=w,
                    n=update_every_n,
                    k=k,
                    m=m,
                    spike_cost=spike_cost,
                    rule=rule,
                    lookahead=lookahead,
                )

                row = {
                    "algorithm": res.algorithm,
                    "size": res.size,
                    "seed": res.seed,
                    "rule": res.extra.get("rule", rule),
                    "lookahead": res.extra.get("lookahead", 0),
                    "found": int(res.found),
                    "steps_taken": res.steps_taken,
                    "total_cost": res.total_cost,
                    "total_runtime_ms": res.total_runtime_ms,
                    "replans": res.replans,
                    "updates": res.updates,
                    "total_nodes_expanded": res.total_nodes_expanded,
                    "total_replan_time_ms": res.total_replan_time_ms,
                    "w": res.extra.get("w", ""),
                    "N": res.extra["N"],
                    "k": res.extra["k"],
                    "m": res.extra["m"],
                    "spike_cost": res.extra["spike_cost"],
                }
                rows.append(row)

                print(
                    f"{algo:7} {res.size:7} seed={seed:2} rule={rule} "
                    f"found={res.found} cost={res.total_cost:.1f} steps={res.steps_taken:3} "
                    f"replans={res.replans:2} exp={res.total_nodes_expanded:6} "
                    f"replan_time={res.total_replan_time_ms:8.2f}ms "
                    f"total_time={res.total_runtime_ms:8.2f}ms"
                )

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")


# Reads the dynamic results CSV, prints a summary of successful runs,
# and saves the summary to a second CSV file.
def summarise_dynamic(csv_path="dynamic_results.csv"):
    df = pd.read_csv(csv_path)
    df_ok = df[df["found"] == 1].copy()

    group_cols = ["algorithm", "size"]
    if "rule" in df_ok.columns:
        group_cols.append("rule")
    if "lookahead" in df_ok.columns and df_ok["lookahead"].notna().any():
        group_cols.append("lookahead")

    summary = (
        df_ok
        .groupby(group_cols, as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_cost=("total_cost", "mean"),
            std_cost=("total_cost", "std"),
            mean_steps=("steps_taken", "mean"),
            std_steps=("steps_taken", "std"),
            mean_replans=("replans", "mean"),
            mean_expanded=("total_nodes_expanded", "mean"),
            std_expanded=("total_nodes_expanded", "std"),
            mean_replan_ms=("total_replan_time_ms", "mean"),
            std_replan_ms=("total_replan_time_ms", "std"),
            mean_total_ms=("total_runtime_ms", "mean"),
            std_total_ms=("total_runtime_ms", "std"),
        )
        .sort_values(group_cols)
    )

    for col in [
        "mean_cost", "std_cost",
        "mean_steps", "std_steps",
        "mean_replans",
        "mean_expanded", "std_expanded",
        "mean_replan_ms", "std_replan_ms",
        "mean_total_ms", "std_total_ms",
    ]:
        summary[col] = summary[col].round(3)

    print("\n=== Dynamic Summary (successful runs only) ===")
    print(summary.to_string(index=False))

    out_summary = csv_path.replace(".csv", "_summary.csv")
    summary.to_csv(out_summary, index=False)
    print(f"\nSaved: {out_summary}")


# Runs both dynamic experiment modes used in the project and summarises their results.
if __name__ == "__main__":
    specs = [SMALL, MEDIUM, LARGE]
    seeds = list(range(10))
    algos = ["UCS", "A*", "wA*", "GBFS", "D*Lite"]

    local_csv = "../results/local_cost_spiking/results_local_cost_spiking.csv"
    path_ahead_csv = "../results/path_ahead_spiking/path_ahead_spiking.csv"

    run_dynamic_batch(
        specs,
        seeds,
        algos,
        out_csv=local_csv,
        w=2.0,
        update_every_n=10,
        k=5,
        m=5,
        spike_cost=50,
        rule="Local Cost Spiking",
    )
    summarise_dynamic(local_csv)

    run_dynamic_batch(
        specs,
        seeds,
        algos,
        out_csv=path_ahead_csv,
        w=2.0,
        update_every_n=10,
        k=5,
        m=5,
        spike_cost=50,
        rule="Path Ahead Spiking",
        lookahead=10,
    )
    summarise_dynamic(path_ahead_csv)