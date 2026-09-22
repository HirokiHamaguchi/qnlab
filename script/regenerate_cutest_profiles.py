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
    "NTRQN",
    "NTRQN-MS",
    "NTRQN-SP",
    "Line",
    "Line-MS",
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
        ("NTRQN", "NTRQN-MS", "NTRQN-SP", "NTRQN-Restart", "NTRQN-MS-Restart"),
    ),
    "eps_nominal": (
        64,
        tuple(range(5)),
        (1e-2,),
        ("NTRQN", "NTRQN-MS", "NTRQN-SP", "NTRQN-Restart", "NTRQN-MS-Restart"),
    ),
    "eps_over": (
        64,
        tuple(range(5)),
        (1e-2,),
        ("NTRQN", "NTRQN-MS", "NTRQN-SP", "NTRQN-Restart", "NTRQN-MS-Restart"),
    ),
}
SCENARIO_ALIASES = {"eps_nominal": "function_only"}


def load_problem_sets() -> dict[int, tuple[str, ...]]:
    data = json.loads(PROBLEM_LIST.read_text(encoding="utf-8"))["valid_problems"]
    return {
        precision: tuple(data[f"precision_{precision}"])
        for precision in (16, 32, 64)
    }


def first_successful_call(path: Path, tolerance: float) -> float:
    with np.load(path) as data:
        calls = np.asarray(data["calls"], dtype=int)
        gnorms = np.asarray(data["gnorms"], dtype=float)
    reached = np.flatnonzero(gnorms <= tolerance)
    return float(max(1, int(calls[reached[0]]))) if reached.size else np.inf


def load_calls(
    scenario: str,
    precision: int,
    seeds: tuple[int, ...],
    tolerance: float,
    methods: tuple[str, ...],
    problem_sets: dict[int, tuple[str, ...]],
) -> np.ndarray:
    source_scenario = SCENARIO_ALIASES.get(scenario, scenario)
    instances = [(seed, problem) for seed in seeds for problem in problem_sets[precision]]
    calls = np.full((len(methods), len(instances)), np.inf)
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
            calls[method_index, instance_index] = first_successful_call(
                path, tolerance
            )
    return calls


def output_path(scenario: str, precision: int, tolerance: float) -> Path:
    tolerance_name = f"{tolerance:.0e}".replace("+", "")
    if scenario.startswith("float"):
        stem = f"precision{precision}"
    elif scenario == "joint_noise":
        stem = "noise0.001"
    else:
        stem = scenario
    return OUTPUT_ROOT / f"_pp_{stem}_gtol{tolerance_name}.pdf"


def draw_profile(methods: tuple[str, ...], calls: np.ndarray, target: Path) -> None:
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
        calls.T,
        linestyle=[LINE_STYLES[name] for name in methods],
        colors=[COLORS[name] for name in methods],
        thetaMax=10.0,
        markersize=6,
        markevery=[0],
        linewidth=2.2,
    )
    ax.set_xlabel(r"Performance Ratio $\tau$", fontsize=18)
    ax.set_ylabel(r"Proportion of Test Instances Solved $\rho_s(\tau)$", fontsize=18)
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


def main() -> None:
    problem_sets = load_problem_sets()
    for scenario, (precision, seeds, tolerances, methods) in SCENARIOS.items():
        for tolerance in tolerances:
            calls = load_calls(
                scenario, precision, seeds, tolerance, methods, problem_sets
            )
            draw_profile(methods, calls, output_path(scenario, precision, tolerance))

    diagnostic_profiles = (
        (
            "function_only_restart",
            ("NTRQN", "NTRQN-MS", "NTRQN-Restart", "NTRQN-MS-Restart"),
        ),
        (
            "ntqn_termination",
            ("NTQN", "NTQN-Default-Termination"),
        ),
    )
    for stem, methods in diagnostic_profiles:
        calls = load_calls(
            "function_only", 64, tuple(range(5)), 1e-2, methods, problem_sets
        )
        draw_profile(methods, calls, OUTPUT_ROOT / f"_pp_{stem}_gtol1e-02.pdf")


if __name__ == "__main__":
    main()
