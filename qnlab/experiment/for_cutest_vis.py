from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qnlab.experiment.for_cutest_run import CUTEstTask, load_npz
from qnlab.experiment.profile import data_profile, performance_profile
from qnlab.experiment.vis import vis
from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.cutest_noised import CUTEstNoisedProblem
from qnlab.util.callback import Callback
from qnlab.util.method import Method, get_box_methods, get_methods

INDIVIDUAL_PLOT_OUTPUT_DIR = Path("doc/imgs/compare/individual")


def _individual_plot_output_path(
    prob_name: str,
    precision: int,
    noise: np.float64,
    boxed: bool,
    metric: str = "calls",
) -> Path:
    """Return the extension-free output path expected by ``vis``."""
    constraint = "boxed" if boxed else "unboxed"
    precision_noise = f"precision{precision}" if noise == 0 else f"noise{noise}"
    if metric == "time":
        return (
            INDIVIDUAL_PLOT_OUTPUT_DIR
            / constraint
            / "time"
            / precision_noise
            / prob_name
        )
    return INDIVIDUAL_PLOT_OUTPUT_DIR / constraint / precision_noise / prob_name


def individual_plot(
    problems: list[str],
    methods: list[tuple[Method, dict]],
    precision: int,
    noise: np.float64,
    boxed: bool = False,
    x_axis: Literal["calls", "iterations", "time"] = "calls",
    result_subdir: str | None = None,
):
    labels = [method.label for method, _ in methods]
    _, color_palette, line_styles = get_box_methods() if boxed else get_methods()
    for prob_name in problems:
        if noise > 0.0:
            prob: CUTEstQNProblem = CUTEstNoisedProblem(
                prob_name,
                precision=precision,
                function_noise=noise,
                gradient_noise=noise,
            )
        else:
            prob = CUTEstQNProblem(prob_name, precision=precision)
        callbacks: list[Callback] = []
        for method, option in methods:
            task = CUTEstTask(
                prob_name,
                method,
                option,
                precision,
                function_noise=noise,
                gradient_noise=noise,
                boxed=boxed,
            )
            callbacks.append(load_npz(task, result_subdir=result_subdir))
        output_path = _individual_plot_output_path(
            prob_name,
            precision,
            noise,
            boxed,
            metric="time" if x_axis == "time" else "calls",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis(
            prob,
            callbacks,
            labels,
            prob_name,
            only_grad=False,
            only_plot=True,
            pdf_path=str(output_path),
            x_axis=x_axis,
            color_palette=color_palette,
            line_styles=line_styles,
        )
        print(f"Saved figure to {output_path}.pdf")


def generate_output_path(
    precision: int,
    noise: np.float64,
    gtol: float,
    boxed: bool = False,
    metric: str = "calls",
) -> Path:
    """Generate output file path for the performance profile plot."""
    output_dir = Path("doc/imgs/compare")
    precision_noise = f"precision{precision}" if noise == 0 else f"noise{noise}"
    gtol_filename = f"{gtol:.0e}".replace("+", "")
    prefix = "_pp_boxed" if boxed else "_pp"
    if metric == "time":
        prefix += "_time"
    return output_dir / f"{prefix}_{precision_noise}_gtol{gtol_filename}.pdf"


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
    boxed: bool = False,
    metric: str = "calls",
    output_path: Path | None = None,
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

    if output_path is None:
        output_path = generate_output_path(precision, noise, gtol, boxed, metric)
    fig_size = generate_fig_size(noise)

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


def draw_data_profile(
    alg_names: list[str],
    callsM: np.ndarray,
    dimensions: np.ndarray,
    color_palette: dict[str, str],
    line_styles: dict[str, str],
    output_path: Path,
    alpha_max: float | None = None,
) -> None:
    """Draw a data profile normalized by each problem's dimension plus one."""
    sns.set_style("whitegrid")
    colors = [color_palette.get(name, "black") for name in alg_names]
    styles = [line_styles.get(name, "o-") for name in alg_names]
    fig, ax = plt.subplots(figsize=(7, 5))
    data_profile(
        callsM.T,
        dimensions,
        linestyle=styles,
        colors=colors,
        alpha_max=alpha_max,
        markersize=6,
        markevery=[0],
        linewidth=2.2,
    )
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6, color="gray")
    ax.set_axisbelow(True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved figure to {output_path}")
    plt.show()
    plt.close()
