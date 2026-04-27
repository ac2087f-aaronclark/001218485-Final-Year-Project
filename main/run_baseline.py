# Runs the baseline experiment batch and then creates
# a simple summary table from the generated CSV results.

import pandas as pd

from maze.grid import SMALL, MEDIUM, LARGE
from experiments.baseline_experiment import run_batch


# Reads the baseline results CSV produced in experiment
# builds a grouped summary table, prints it, and saves it to a new CSV.
def summarise_results(csv_path: str = "../results/baseline/baseline_results.csv") -> None:
    # Load the full baseline results file in panda dataframe
    df = pd.read_csv(csv_path)

    # Failsafe Keep only runs where a path to the goal was found.
    df_ok = df[df["found"] == 1].copy()

    # Group results by algorithm and grid size, then compute the main averages.
    #std refers to standard deviation
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

    # Round values so the printed table and saved CSV stay readable.
    summary["mean_cost"] = summary["mean_cost"].round(2)
    summary["std_cost"] = summary["std_cost"].round(2)
    summary["mean_expanded"] = summary["mean_expanded"].round(1)
    summary["std_expanded"] = summary["std_expanded"].round(1)
    summary["mean_ms"] = summary["mean_ms"].round(3)
    summary["std_ms"] = summary["std_ms"].round(3)
    summary["mean_path_len"] = summary["mean_path_len"].round(1)

    print("\n=== Summary (successful runs only) ===")
    print(summary.to_string(index=False))

    # Save the grouped summary as a second CSV file.
    summary.to_csv("../results/baseline/baseline_results_summary.csv", index=False)
    print("\nSaved: ../results/baseline/baseline_results_summary.csv")


# Runs the full baseline batch using project settings. Can increase amount of runs
if __name__ == "__main__":
    specs = [SMALL, MEDIUM, LARGE]
    seeds = list(range(10))
    algos = ["UCS", "A*", "wA*", "GBFS", "D*Lite"]
#this where it saves batchline results from experiment
    run_batch(specs, seeds, algos, w=2.0, out_csv="../results/baseline/baseline_results.csv")
    summarise_results("../results/baseline/baseline_results.csv")