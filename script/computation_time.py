import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qnlab-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qnlab.problem.ill_quadratic import IllQuadraticProblem
from qnlab.solver.qn import qn
from qnlab.util.method import COLORS, get_methods

# Configuration
n = 10000
m = 10
MI = 100
num_runs = 100

(methods, *_) = get_methods(m=m, MI=MI)
methods = [entry for entry in methods if entry[0].label != "NTRQN-MS-SP"]


def display_name(method_name: str) -> str:
    if method_name == "NTRQN-Restart":
        return "Ours-R"
    return method_name.replace("NTRQN", "Ours")


def internal_name(method_name: str) -> str:
    if method_name == "Ours-R":
        return "NTRQN-Restart"
    return method_name.replace("Ours", "NTRQN")


def run_benchmark() -> pd.DataFrame:
    results = {
        "method": [],
        "run": [],
        "time": [],
        "num_of_calls": [],
        "fx": [],
    }

    print(f"Running timing benchmark on IllQuadraticProblem(n={n})")
    print(f"Each method will be run {num_runs} times\n")

    # Run benchmarks
    for method, option in methods:
        method_name = method.label
        print(f"Testing {method_name}...")

        warmup_count = 3
        for run_idx in range(num_runs + warmup_count):
            prob = IllQuadraticProblem(n=n)

            start_time = time.perf_counter()
            _info, fx, _x_opt = qn(prob, method, option, callback=None, verbose=False)
            elapsed_time = time.perf_counter() - start_time

            print(
                f"  Run {run_idx + 1}/{num_runs + warmup_count}: {elapsed_time:.6f} seconds"
            )

            if run_idx < warmup_count:
                continue

            results["method"].append(display_name(method_name))
            results["run"].append(run_idx - warmup_count + 1)
            results["time"].append(elapsed_time)
            results["num_of_calls"].append(prob.count_calls())
            results["fx"].append(fx)

        print()

    # Create DataFrame
    df = pd.DataFrame(results)
    print("=" * 80)
    print("Results collected successfully!")
    assert len(set(results["num_of_calls"])) <= len(methods), set(
        results["num_of_calls"]
    )
    assert len(set(results["fx"])) <= len(methods)

    return df


def summarize_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = (
        df.groupby("method")["time"].agg(["mean", "std", "min", "max"]).reset_index()
    )
    stats.columns = [
        "Method",
        "Mean Time (s)",
        "Std Dev (s)",
        "Min Time (s)",
        "Max Time (s)",
    ]
    stats = stats.sort_values("Mean Time (s)")

    table_data = (
        df.groupby("method")
        .agg({"fx": "first", "num_of_calls": "first", "time": "mean"})
        .reset_index()
    )
    table_data = table_data.sort_values("time")

    return stats, table_data


def vis_benchmark(stats: pd.DataFrame):
    # Set style for publication
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    methods_list = stats["Method"].tolist()
    methods_list = [display_name(name) for name in methods_list]
    means = stats["Mean Time (s)"].tolist()
    stds = stats["Std Dev (s)"].tolist()

    x_pos = np.arange(len(methods_list))
    colors = [COLORS[internal_name(name)] for name in methods_list]

    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=4,
        alpha=0.85,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
    )

    ax.set_ylabel(r"Execution Time (seconds)", fontsize=13, fontweight="normal")
    ax.set_title(
        rf"Execution Time Comparison $(n={n})$, {num_runs} Runs",
        fontsize=14,
        fontweight="normal",
        pad=12,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods_list, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.4, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std * 1.2,
            f"${mean:.3f}$",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.ylim(0, max(bar.get_height() + std * 1.2 for bar, std in zip(bars, stds)) * 1.1)
    plt.tight_layout()

    return fig, methods_list


def generate_latex_table(methods_list: list[str], table_data: pd.DataFrame):
    num_methods = len(methods_list)

    latex_table = (
        r"""\begin{table}[t]
    \centering
    \small
    \setlength{\tabcolsep}{3.5pt}
    \caption{The number of oracle calls and final observed objective values for the experiment in \cref{sec:comp_time}.}
    \label{tab:time_results}
    \begin{tabular}{l"""
        + "r" * num_methods
        + r"""}
    \toprule
                                & """
        + " & ".join(r"\texttt{" + x + "}" for x in methods_list)
        + r""" \\
    \midrule
    """
    )

    calls = [str(int(x)) for x in table_data["num_of_calls"].tolist()]
    latex_table += r"        \# Calls & " + " & ".join(calls) + r" \\" + "\n"
    fx_values = [f"{x:.2f}" for x in table_data["fx"].tolist()]
    latex_table += (
        r"    $\overline{f}(x_{k_{\max}})$ & " + " & ".join(fx_values) + r" \\" + "\n"
    )

    latex_table += (
        r"""    \bottomrule
    \end{tabular}"""
        + "\n\\end{table}"
    )

    return latex_table


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the figure and table from the saved timing data.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    repo_root = Path(__file__).parent.parent.resolve()
    assert (repo_root / "doc" / "main" / "check").exists()
    data_path = repo_root / "data" / "computation_time.csv"

    if args.plot_only:
        df = pd.read_csv(data_path)
        df = df[df["method"] != "Line-MS"]
        print(f"Loaded timing data from {data_path.relative_to(repo_root)}")
    else:
        df = run_benchmark()
        df.to_csv(data_path, index=False)
        print(f"Saved timing data to {data_path.relative_to(repo_root)}")

    stats, table_data = summarize_results(df)
    fig, methods_list = vis_benchmark(stats)

    # Save as PDF
    pdf_path = repo_root / "doc" / "imgs" / "for_paper" / "time.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved figure to {pdf_path}")
    plt.close(fig)

    latex_table = generate_latex_table(methods_list, table_data)
    table_path = repo_root / "doc" / "main" / "check" / "time_results_table.tex"
    table_path.write_text(latex_table + "\n", encoding="utf-8", newline="\n")
    print(f"Saved LaTeX table to {table_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
