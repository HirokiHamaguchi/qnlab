import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qnlab.problem.ill_quadratic import IllQuadraticProblem
from qnlab.solver.qn import qn
from qnlab.util.method import get_methods

# Configuration
n = 10000
m = 10
MI = 100
num_runs = 100

(methods, *_) = get_methods(m=m, MI=MI)


def run_benchmark():
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
            info, fx, x_opt = qn(prob, method, option, callback=None, verbose=False)
            elapsed_time = time.perf_counter() - start_time

            print(
                f"  Run {run_idx + 1}/{num_runs + warmup_count}: {elapsed_time:.6f} seconds"
            )

            if run_idx < warmup_count:
                continue

            results["method"].append(method_name.replace("Hamaguchi", "Ours"))
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

    return df, stats, table_data


def vis_benchmark(stats: pd.DataFrame):
    # Set style for publication
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    methods_list = stats["Method"].tolist()
    methods_list = [m.replace("Hamaguchi", "Ours") for m in methods_list]
    means = stats["Mean Time (s)"].tolist()
    stds = stats["Std Dev (s)"].tolist()

    x_pos = np.arange(len(methods_list))
    colors = sns.color_palette("viridis", len(methods_list))

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

    ax.set_ylabel(r"Execution Time (seconds)", fontsize=11, fontweight="normal")
    ax.set_title(
        rf"Execution Time Comparison $(n={n})$, {num_runs} Runs",
        fontsize=12,
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
            fontsize=9,
        )

    plt.ylim(0, max(bar.get_height() + std * 1.2 for bar, std in zip(bars, stds)) * 1.1)
    plt.tight_layout()

    # Create directory if it doesn't exist
    output_path = Path("../doc/imgs/for_paper")
    output_path.mkdir(parents=True, exist_ok=True)

    return fig, output_path, methods_list


def generate_latex_table(methods_list: list[str], table_data: pd.DataFrame):
    num_methods = len(methods_list)

    latex_table = (
        r"""\begin{table}[ht]
        \centering
        \caption{The number of oracle calls and final observed objective values for the experiment in \cref{sec:comp_time}.}
        \label{tab:time_results}
        \begin{tabular}{l"""
        + "r" * num_methods
        + r"""}
            \toprule
                                        & """
        + " & ".join(map(lambda x: r"\texttt{" + x + "}", methods_list))
        + r""" \\
            \midrule
    """
    )

    calls = [str(int(x)) for x in table_data["num_of_calls"].tolist()]
    latex_table += r"        \# Calls & " + " & ".join(calls) + r" \\" + "\n"
    fx_values = [f"{x:.2f}" for x in table_data["fx"].tolist()]
    latex_table += (
        r"        $\overline{f}(x_{k_{\max}})$ & "
        + " & ".join(fx_values)
        + r" \\"
        + "\n"
    )

    latex_table += r"""        \bottomrule
        \end{tabular}
    \end{table}
    """

    return latex_table
