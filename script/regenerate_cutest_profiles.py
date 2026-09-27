"""Regenerate CUTEst performance profiles from saved NPZ results only."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qnlab-matplotlib")
)

import matplotlib.pyplot as plt
import numpy as np

from qnlab.experiment.profile import performance_profile
from qnlab.util.method import COLORS, LINE_STYLES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "data" / "temp"
OUTPUT_ROOT = REPOSITORY_ROOT / "doc" / "imgs" / "compare"
PROBLEM_LIST = REPOSITORY_ROOT / "data" / "CUTEst" / "valid_problems.json"

STANDARD_METHODS = (
    "Ours",
    "Ours-Heuristic",
    "Line",
    "Reg",
    "Reg-Sec",
    "SciPy",
    "NTQN",
    "ASTR1-Adagrad",
)

SCENARIOS = {
    "float64": (64, (0,), (1e-1, 1e-3, 1e-5), STANDARD_METHODS),
    "float32": (32, (0,), (1e-1, 1e-3, 1e-5), STANDARD_METHODS),
    "float16": (16, (0,), (1e-1, 1e-3, 1e-5), STANDARD_METHODS),
    "function_only": (64, tuple(range(5)), (1e-2,), STANDARD_METHODS),
    "joint_noise": (64, tuple(range(5)), (1e-2,), STANDARD_METHODS),
    "eps_under": (
        64,
        tuple(range(5)),
        (1e-2,),
        ("Ours", "Ours-Heuristic"),
    ),
    "eps_nominal": (
        64,
        tuple(range(5)),
        (1e-2,),
        ("Ours", "Ours-Heuristic"),
    ),
    "eps_over": (
        64,
        tuple(range(5)),
        (1e-2,),
        ("Ours", "Ours-Heuristic"),
    ),
}

SCENARIO_ALIASES = {"eps_nominal": "function_only"}

PLOT_COLORS = COLORS | {
    "Ours": COLORS["NTRQN-MS"],
    "Ours-Heuristic": COLORS["NTRQN-Restart"],
}
PLOT_LINE_STYLES = LINE_STYLES | {
    "Ours": LINE_STYLES["NTRQN-MS"],
    "Ours-Heuristic": LINE_STYLES["NTRQN-Restart"],
}


def load_problem_sets() -> dict[int, tuple[str, ...]]:
    data = json.loads(PROBLEM_LIST.read_text(encoding="utf-8"))["valid_problems"]
    return {
        precision: tuple(data[f"precision_{precision}"]) for precision in (16, 32, 64)
    }


def first_successful_call(path: Path, tolerance: float) -> float:
    with np.load(path) as data:
        calls = np.asarray(data["calls"], dtype=int)
        gnorms = np.asarray(data["gnorms"], dtype=float)

    reached = np.flatnonzero(gnorms <= tolerance)
    return float(max(1, int(calls[reached[0]]))) if reached.size else np.inf


def first_successful_time(path: Path, tolerance: float) -> float:
    with np.load(path) as data:
        times = np.asarray(data["times"], dtype=float)
        gnorms = np.asarray(data["gnorms"], dtype=float)

    reached = np.flatnonzero(gnorms <= tolerance)
    return float(times[reached[0]]) if reached.size else np.inf


def load_metric(
    scenario: str,
    precision: int,
    seeds: tuple[int, ...],
    tolerance: float,
    methods: tuple[str, ...],
    problem_sets: dict[int, tuple[str, ...]],
    metric: str,
) -> np.ndarray:
    source_scenario = SCENARIO_ALIASES.get(scenario, scenario)
    instances = [
        (seed, problem) for seed in seeds for problem in problem_sets[precision]
    ]

    values = np.full((len(methods), len(instances)), np.inf)

    for method_index, method in enumerate(methods):
        for instance_index, (seed, problem) in enumerate(instances):
            path = (
                RESULT_ROOT
                / source_scenario
                / f"seed_{seed}"
                / problem
                / f"{method}.npz"
            )

            if not path.is_file():
                raise FileNotFoundError(path)

            if metric == "calls":
                value = first_successful_call(path, tolerance)
            elif metric == "time":
                value = first_successful_time(path, tolerance)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            values[method_index, instance_index] = value

    return values


def output_path(
    scenario: str,
    precision: int,
    tolerance: float,
    metric: str = "calls",
) -> Path:
    tolerance_name = f"{tolerance:.0e}".replace("+", "")

    if scenario.startswith("float"):
        stem = f"precision{precision}"
    elif scenario == "joint_noise":
        stem = "noise0.001"
    else:
        stem = scenario

    metric_suffix = "" if metric == "calls" else "_time"

    return OUTPUT_ROOT / f"_pp_{stem}_gtol{tolerance_name}{metric_suffix}.pdf"


def draw_profile(
    methods: tuple[str, ...],
    values: np.ndarray,
    target: Path,
    colors: dict[str, object] = PLOT_COLORS,
    line_styles: dict[str, str] = PLOT_LINE_STYLES,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "font.family": "serif",
            "font.size": 20,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))

    performance_profile(
        values.T,
        linestyle=[line_styles[name] for name in methods],
        colors=[colors[name] for name in methods],
        thetaMax=10.0,
        markersize=6,
        markevery=[0],
        linewidth=2.2,
    )

    ax.set_xlabel(r"Performance Ratio $\tau$", fontsize=18)
    ax.set_ylabel(
        r"Proportion of Test Instances Solved $\rho_s(\tau)$",
        fontsize=18,
    )
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6, color="gray")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    fig.tight_layout()

    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved figure to {target}")


def draw_time_to_solution(
    methods: tuple[str, ...],
    times: np.ndarray,
    target: Path,
) -> None:
    """Plot the fraction attaining the tolerance against absolute runtime."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "font.family": "serif",
            "font.size": 20,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    instance_count = times.shape[1]
    positive_times = times[np.isfinite(times) & (times > 0)]
    if positive_times.size == 0:
        raise ValueError("No positive finite runtimes to plot")
    left = positive_times.min() * 0.95
    right = positive_times.max() * 1.05
    for method, method_times in zip(methods, times, strict=True):
        finite = np.sort(method_times[np.isfinite(method_times)])
        fractions = np.arange(1, finite.size + 1) / instance_count
        ax.step(
            np.r_[left, np.maximum(finite, left), right],
            np.r_[0.0, fractions, finite.size / instance_count],
            PLOT_LINE_STYLES[method],
            color=PLOT_COLORS[method],
            where="post",
            markersize=6,
            markevery=[1] if finite.size else [],
            linewidth=2.2,
        )
    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(r"Runtime (s)", fontsize=18)
    ax.set_ylabel(r"Proportion of Test Problems Solved", fontsize=18)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6, color="gray")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figure to {target}")


def main() -> None:
    problem_sets = load_problem_sets()

    for scenario, (precision, seeds, tolerances, methods) in SCENARIOS.items():
        for tolerance in tolerances:
            # Function-call performance profile
            calls = load_metric(
                scenario,
                precision,
                seeds,
                tolerance,
                methods,
                problem_sets,
                metric="calls",
            )
            draw_profile(
                methods,
                calls,
                output_path(
                    scenario,
                    precision,
                    tolerance,
                    metric="calls",
                ),
            )

            if scenario == "float32" and tolerance == 1e-3:
                times = load_metric(
                    scenario,
                    precision,
                    seeds,
                    tolerance,
                    methods,
                    problem_sets,
                    metric="time",
                )
                draw_time_to_solution(
                    methods,
                    times,
                    OUTPUT_ROOT / "_time_precision32_gtol1e-03.pdf",
                )

    sensitivity_methods: list[str] = []
    sensitivity_values: list[np.ndarray] = []
    sensitivity_colors: dict[str, object] = {}
    sensitivity_line_styles: dict[str, str] = {}
    setting_styles = {
        "eps_under": ("underestimated", "--"),
        "eps_nominal": ("nominal", "-"),
        "eps_over": ("overestimated", ":"),
    }
    for scenario, (setting, style) in setting_styles.items():
        precision, seeds, tolerances, methods = SCENARIOS[scenario]
        values = load_metric(
            scenario,
            precision,
            seeds,
            tolerances[0],
            methods,
            problem_sets,
            metric="calls",
        )
        for method, method_values in zip(methods, values, strict=True):
            label = f"{method} ({setting})"
            marker = "o" if method == "Ours" else "s"
            sensitivity_methods.append(label)
            sensitivity_values.append(method_values)
            sensitivity_colors[label] = PLOT_COLORS[method]
            sensitivity_line_styles[label] = marker + style
    draw_profile(
        tuple(sensitivity_methods),
        np.asarray(sensitivity_values),
        OUTPUT_ROOT / "_pp_error_bound_sensitivity_gtol1e-02.pdf",
        sensitivity_colors,
        sensitivity_line_styles,
    )

    diagnostic_profiles = (
        (
            "function_only_restart",
            ("Ours", "Ours-Heuristic"),
        ),
        (
            "ntqn_termination",
            ("NTQN", "NTQN-Default-Termination"),
        ),
    )

    for stem, methods in diagnostic_profiles:
        calls = load_metric(
            "function_only",
            64,
            tuple(range(5)),
            1e-2,
            methods,
            problem_sets,
            metric="calls",
        )
        draw_profile(
            methods,
            calls,
            OUTPUT_ROOT / f"_pp_{stem}_gtol1e-02.pdf",
        )

        times = load_metric(
            "function_only",
            64,
            tuple(range(5)),
            1e-2,
            methods,
            problem_sets,
            metric="time",
        )
        draw_profile(
            methods,
            times,
            OUTPUT_ROOT / f"_pp_{stem}_gtol1e-02_time.pdf",
        )


if __name__ == "__main__":
    main()
