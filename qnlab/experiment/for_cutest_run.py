import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, TypeAlias
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qnlab.experiment.profile import performance_profile
from qnlab.experiment.vis import vis
from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.cutest_noised import CUTEstNoisedProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback, CallbackTimeoutError
from qnlab.util.method import Method

task_type: TypeAlias = Tuple[str, Method, dict, int, np.float64]


def _build_cutest_problem(
    prob_name: str,
    precision: int,
    noise: np.float64,
) -> CUTEstQNProblem:
    if noise > 0.0:
        return CUTEstNoisedProblem(prob_name, precision=precision, noise=noise)
    return CUTEstQNProblem(prob_name, precision=precision)


def get_file_path(task: task_type) -> str:
    prob_name, method, _option, precision, noise = task
    prob_type = "noisy" if noise > 0 else str(precision)
    folder = Path(os.path.dirname(__file__)).parent.parent / "data" / "temp"
    return str(folder / prob_type / prob_name / f"{method.label}.npz")


def load_npz(task: task_type, verbose: bool = True) -> Callback:
    """Load stored callback data for a task."""
    file_path = get_file_path(task)
    if not os.path.exists(file_path):
        if verbose:
            warnings.warn(f"File not found: {file_path}, returning empty Callback.")
        callback = Callback()
        return callback
    try:
        with np.load(file_path) as data:
            callback = Callback()
            callback.calls = data["calls"].astype(int)
            callback.fxs = data["fxs"]
            callback.gnorms = data["gnorms"]
        assert len(callback.calls) == len(callback.fxs) == len(callback.gnorms)
        assert len(callback.calls) > 0
    except (EOFError, BadZipFile) as e:
        print(file_path)
        raise e
    callback.xs = [np.zeros(0) for _ in range(len(callback.calls))]
    return callback


def save_npz(task: task_type, callback: Callback) -> None:
    """Persist callback data for a task."""
    file_path = get_file_path(task)
    np.savez_compressed(
        file_path,
        calls=callback.calls,
        fxs=callback.fxs,
        gnorms=callback.gnorms,
    )


def solveProblemWithTimeout(task: task_type, time_limit: Optional[float] = None):
    prob_name, method, option, precision, noise = task
    file_path = get_file_path(task)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    prob = _build_cutest_problem(prob_name, precision, noise)
    callback = Callback(time_limit=time_limit)
    try:
        print(f"▶ Running: {file_path}")
        qn(prob, method, option, callback)
        print(f"✓ {prob_name} with {method.label}")
        save_npz(task, callback)
    except CallbackTimeoutError as e:
        print(f"⏱ {prob_name} with {method.label}: {str(e)}")
        if time_limit is not None and time_limit < 600:
            print("⚠ Not saving results for time limit less than 600 seconds.")
        else:
            save_npz(task, callback)
    except Exception as e:
        print(f"✗ {prob_name} with {method.label}: {str(e)}")


def run(
    problems: list[str],
    methods: list[Tuple[Method, dict]],
    precision: int,
    noise: np.float64,
    TL: int,
):
    # Prepare all tasks
    tasks: List[task_type] = []
    for problem in problems:
        for method, option in methods:
            task = (problem, method, option, precision, noise)
            if len(load_npz(task, False).calls) > 0:
                continue
            tasks.append(task)
    print(f"Total tasks to run: {len(tasks)}")

    errors = []  # collect errors for reporting after all tasks
    for i, task in enumerate(tasks):
        file_path = get_file_path(task)
        try:
            solveProblemWithTimeout(task, TL)
            print(f"[{i + 1}/{len(tasks)}] ✓ Success")
        except Exception as e:
            print(f"[{i + 1}/{len(tasks)}] ⚠ Error: {e}")
            errors.append((file_path, f"Error: {e}"))


def individual_plot(
    problems: list[str],
    methods: list[Tuple[Method, dict]],
    precision: int,
    noise: np.float64,
):
    labels = [method.label for method, _ in methods]
    for prob_name in problems:
        prob = _build_cutest_problem(prob_name, precision, noise)
        callbacks: List[Callback] = []
        for method, option in methods:
            task: task_type = (prob_name, method, option, precision, noise)
            callbacks.append(load_npz(task))
        vis(prob, callbacks, labels, prob_name, only_grad=True, only_plot=True)


def load_results(
    methods: list[Tuple[Method, dict]],
    problems: list[str],
    precision: int,
    noise: np.float64,
    gtol: np.float64,
) -> Tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and aggregate results from saved files.

    Returns:
        alg_names: List of algorithm names
        callsM: (n_algorithms, n_problems) array of function calls
        fxsM: (n_algorithms, n_problems) array of function values
        gnormsM: (n_algorithms, n_problems) array of gradient norms
        problems: Filtered list of problem names (after removing zero rows)
    """
    alg_names = list(method.label for method, _ in methods)
    nAlgorithms = len(alg_names)
    nProbs = len(problems)
    callsM = np.zeros((nAlgorithms, nProbs), dtype=float)
    fxsM = np.zeros((nAlgorithms, nProbs), dtype=float)
    gnormsM = np.zeros((nAlgorithms, nProbs), dtype=float)

    for j, prob_name in enumerate(problems):
        for i, (method, option) in enumerate(methods):
            task: task_type = (prob_name, method, option, precision, noise)
            callback = load_npz(task, False)
            if len(callback.calls) == 0:
                res = (np.inf, np.inf, np.inf)
            else:
                isOk = callback.gnorms <= gtol
                if not np.any(isOk):
                    res = (np.inf, np.inf, np.inf)
                else:
                    idx = np.where(isOk)[0][0]
                    assert callback.calls[idx] >= 0
                    # We take max with 1
                    # Otherwise, performance profile will cause error
                    call_max_1 = max(1, callback.calls[idx])
                    res = (call_max_1, callback.fxs[idx], callback.gnorms[idx])

            callsM[i, j], fxsM[i, j], gnormsM[i, j] = res

    return alg_names, callsM, fxsM, gnormsM, problems


def generate_title(precision: int, noise: np.float64, gtol: float) -> str | None:
    """Generate title string for the performance profile plot."""
    if noise == 0:
        return None

    noise_e = int(np.log10(noise))
    assert np.isclose(noise, 10**noise_e)
    gtol_e = int(np.log10(gtol))
    assert np.isclose(gtol, 10**gtol_e)
    return (
        rf"noise=$10^{{{noise_e}}}$, "
        + r"$\epsilon_{\mathrm{gtol}}="
        + f"10^{{{gtol_e}}}$"
    )


def generate_output_path(precision: int, noise: np.float64, gtol: float) -> Path:
    """Generate output file path for the performance profile plot."""
    output_dir = Path("doc/imgs/compare")
    precision_noise = f"precision{precision}" if noise == 0 else f"noise{noise}"
    gtol_filename = f"{gtol:.0e}".replace("+", "")
    return output_dir / f"_pp_{precision_noise}_gtol{gtol_filename}.pdf"


def generate_fig_size(noise: np.float64) -> tuple[float, float]:
    """Generate figure size based on noise level."""
    return (7, 5.5) if noise > 0 else (7, 5)


def draw_pp(
    alg_names: list[str],
    callsM: np.ndarray,
    color_palette: dict[str, str],
    line_styles: dict[str, str],
    precision: int,
    noise: np.float64,
    gtol: np.float64,
) -> None:
    # Set style for publication quality
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 20,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )

    # Assign colors and line styles based on method names
    colors = [color_palette.get(name, "black") for name in alg_names]
    line_styles_list = [line_styles.get(name, "o-") for name in alg_names]

    output_path = generate_output_path(precision, noise, gtol)
    fig_size = generate_fig_size(noise)
    title = generate_title(precision, noise, gtol)

    # Create figure with proper size for paper
    fig, ax = plt.subplots(figsize=fig_size)

    # Draw performance profiles
    performance_profile(
        callsM.T,
        linestyle=line_styles_list,
        colors=colors,
        thetaMax=10.0,
        markersize=6,
        markevery=[0],
        linewidth=2.2,
    )

    # Customize the plot
    ax = plt.gca()
    ax.set_xlabel(r"Performance Ratio $\tau$", fontsize=18, fontweight="normal")
    ax.set_ylabel(
        r"Proportion of Problems Solved $\rho_s(\tau)$",
        fontsize=18,
        fontweight="normal",
    )

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=25, fontweight="normal", pad=15)

    # Grid styling
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6, color="gray")
    ax.set_axisbelow(True)

    # Improve spine visibility
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()

    # Save figure if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved figure to {output_path}")

    plt.show()
    plt.close()
