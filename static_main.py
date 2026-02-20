# static_main.py
from maze.grid import SMALL, MEDIUM, LARGE
from static_simulation import run_batch

def summarise_results(csv_path: str = "results.csv") -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)

    # keep only successful runs (just in case later you get failures)
    df_ok = df[df["found"] == 1].copy()

    # summary stats per algorithm + size
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

    # round for readability
    summary["mean_cost"] = summary["mean_cost"].round(2)
    summary["std_cost"] = summary["std_cost"].round(2)
    summary["mean_expanded"] = summary["mean_expanded"].round(1)
    summary["std_expanded"] = summary["std_expanded"].round(1)
    summary["mean_ms"] = summary["mean_ms"].round(3)
    summary["std_ms"] = summary["std_ms"].round(3)
    summary["mean_path_len"] = summary["mean_path_len"].round(1)

    print("\n=== Summary (successful runs only) ===")
    print(summary.to_string(index=False))

    # optional: save summary to CSV as well
    summary.to_csv("results_summary.csv", index=False)
    print("\nSaved: results_summary.csv")




if __name__ == "__main__":
    specs = [SMALL, MEDIUM, LARGE]
    seeds = list(range(10))          # 0..9
    algos = ["UCS", "A*", "wA*", "GBFS", "D*Lite"]

    run_batch(specs, seeds, algos, w=2.0, out_csv="results.csv")
    summarise_results("results.csv")