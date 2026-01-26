from pathlib import Path
from typing import List, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qnlab.experiment.for_cutest_run import load_npz
from qnlab.experiment.profile import performance_profile
from qnlab.experiment.vis import vis
from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.cutest_noised import CUTEstNoisedProblem
from qnlab.util.callback import Callback
from qnlab.util.method import Method

task_type: TypeAlias = Tuple[str, Method, dict, int, np.float64]


def individual_plot(
    problems: list[str],
    methods: list[Tuple[Method, dict]],
    precision: int,
    noise: np.float64,
):
    labels = [method.label for method, _ in methods]
    for prob_name in problems:
        if noise > 0.0:
            prob: CUTEstQNProblem = CUTEstNoisedProblem(
                prob_name, precision=precision, noise=noise
            )
        else:
            prob = CUTEstQNProblem(prob_name, precision=precision)
        callbacks: List[Callback] = []
        for method, option in methods:
            task: task_type = (prob_name, method, option, precision, noise)
            callbacks.append(load_npz(task))
        vis(prob, callbacks, labels, prob_name, only_grad=True, only_plot=True)


def generate_title(noise: np.float64, gtol: float) -> str | None:
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
    title = generate_title(noise, gtol)

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
