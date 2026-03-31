"""
This file runs the baseline experiment batch and then creates
a simple summary table from the generated CSV results.
"""

import pandas as pd

from maze.grid import SMALL, MEDIUM, LARGE
from experiments.baseline_experiment import run_batch


# Reads the baseline results CSV, prints a summary of successful runs,
# and saves the summary to a second CSV file.
def summarise_results(csv_path: str = "../results/baseline/baseline_results.csv") -> None:
    df = pd.read_csv(csv_path)

    df_ok = df[df["found"] == 1].copy()

    summary = (
        df_ok
        .groupby(["algorithm", "size"], as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_cost=("total_cost", "mean"),
            std_cost=("total_cost", "std"),
            mean_expanded=("nodes_expanded", "mean"),
            std_expanded=("nodes_expanded", "std"),
            mean_ms=("runtime_ms", "mean"),
            std_ms=("runtime_ms", "std"),
            mean_path_len=("path_len", "mean"),
        )
        .sort_values(["size", "algorithm"])
    )

    summary["mean_cost"] = summary["mean_cost"].round(2)
    summary["std_cost"] = summary["std_cost"].round(2)
    summary["mean_expanded"] = summary["mean_expanded"].round(1)
    summary["std_expanded"] = summary["std_expanded"].round(1)
    summary["mean_ms"] = summary["mean_ms"].round(3)
    summary["std_ms"] = summary["std_ms"].round(3)
    summary["mean_path_len"] = summary["mean_path_len"].round(1)

    print("\n=== Summary (successful runs only) ===")
    print(summary.to_string(index=False))

    summary.to_csv("../results/baseline/baseline_results_summary.csv", index=False)
    print("\nSaved: ../results/baseline/baseline_results_summary.csv")


# Runs the baseline batch using the fixed experiment settings for the project.
if __name__ == "__main__":
    specs = [SMALL, MEDIUM, LARGE]
    seeds = list(range(10))
    algos = ["UCS", "A*", "wA*", "GBFS", "D*Lite"]

    run_batch(specs, seeds, algos, w=2.0, out_csv="../results/baseline/baseline_results.csv")
    summarise_results("../results/baseline/baseline_results.csv")