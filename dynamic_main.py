# dynamic_main.py
import csv
import pandas as pd

from maze.grid import Grid, SMALL, MEDIUM, LARGE
from dynamic_simulation import run_dynamic_episode


def run_dynamic_batch(
    specs,
    seeds,
    algos,
    *,
    out_csv="dynamic_results.csv",
    w=2.0,
    N=10,
    k=5,
    m=5,
    spike_cost=50,
):
    rows = []

    for spec in specs:
        for seed in seeds:
            grid = Grid(spec, seed=seed)

            for algo in algos:
                res = run_dynamic_episode(
                    grid,
                    algo,
                    seed=seed,
                    w=w,
                    N=N,
                    k=k,
                    m=m,
                    spike_cost=spike_cost,
                )

                row = {
                    "algorithm": res.algorithm,
                    "size": res.size,
                    "seed": res.seed,
                    "found": int(res.found),
                    "steps_taken": res.steps_taken,
                    "total_cost": res.total_cost,
                    "total_runtime_ms": res.total_runtime_ms,  # NEW
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
                    f"{algo:7} {res.size:7} seed={seed:2} "
                    f"found={res.found} cost={res.total_cost:.1f} steps={res.steps_taken:3} "
                    f"replans={res.replans:2} exp={res.total_nodes_expanded:5} "
                    f"replan_time={res.total_replan_time_ms:7.2f}ms "
                    f"total_time={res.total_runtime_ms:7.2f}ms"
                )

    # write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_csv}")


def summarise_dynamic(csv_path="dynamic_results.csv"):
    df = pd.read_csv(csv_path)
    df_ok = df[df["found"] == 1].copy()

    summary = (
        df_ok
        .groupby(["algorithm", "size"], as_index=False)
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
            mean_total_ms=("total_runtime_ms", "mean"),  # NEW
            std_total_ms=("total_runtime_ms", "std"),    # NEW
        )
        .sort_values(["size", "algorithm"])
    )

    # round for readability
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

    summary.to_csv("dynamic_results_summary.csv", index=False)
    print("\nSaved: dynamic_results_summary.csv")


if __name__ == "__main__":
    specs = [SMALL, MEDIUM, LARGE]
    seeds = list(range(10))
    algos = ["UCS", "A*", "wA*", "GBFS", "D*Lite"]

    run_dynamic_batch(
        specs,
        seeds,
        algos,
        out_csv="dynamic_results.csv",
        w=2.0,
        N=10,
        k=5,
        m=5,
        spike_cost=50,
    )

    summarise_dynamic("dynamic_results.csv")