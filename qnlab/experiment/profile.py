"""
The code is based on `perfprof` from the MATLAB Guide
by D. J. Higham and N. J. Higham:
https://github.com/higham/matlab-guide-3ed/blob/master/perfprof.m
"""

from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _compute_theta_max(data, minVals):
    assert np.all(minVals > 0)
    tmax = np.max(data, axis=1, initial=0, where=(data < np.inf))
    thetaMax = np.max(tmax / minVals, initial=1.01)
    return thetaMax


def _compute_theta(col, minVals):
    """
    Performance ratios for an individual solver against the vector of minimum values.
    Problems that are not solved by any algorithm have their ratios set to Inf.
    """
    assert np.all(minVals > 0)
    th = np.full(np.shape(col), np.inf)
    valid = minVals < np.inf
    th[valid] = col[valid] / minVals[valid]
    return th


def _make_staircase(col, m, thetaMax, tol):
    """
    Assemble staircase (x, y) pairs.
    col : "column" of theta values
    m : number of problems
    thetaMax : maximum value of theta for endpoint clamping
    tol : theta tolerance for endpoint clamping
    """
    theta, counts = np.unique(col, return_counts=True)
    prob = np.cumsum(counts) / m

    # Ensure endpoints plotted correctly
    if theta[0] >= 1 + tol:
        theta = np.append(1, theta)
        prob = np.append(0, prob)
    if theta[-1] < thetaMax - tol:
        theta = np.append(theta, thetaMax)
        prob = np.append(prob, prob[-1])

    return theta, prob


def performance_profile(
    data,
    linestyle,
    colors,
    thetaMax=None,
    tol=np.sqrt(np.finfo(np.double).eps),
    **kwargs,
) -> Any:
    """
    Performance profile for the input data.

    Parameters
    ----------
    data : Array of timings/errors to plot.
           M-by-N matrix where data[i, j] > 0 measures the performance of the
           j-th solver on the i-th problem, with smaller values denoting "better".

    linestyle : List of line specs, e.g., ['o-r', '-.g']

    thetaMax : Maximum value of theta shown on the x-axis.
            Defaults to max(tm, 1.01), where tm is the largest finite performance ratio.

    tol : Tolerance for endpoint clamping.
          Defaults to sqrt(eps), where eps is the double precision machine accuracy.

    **kwargs : Optional keyword args to be forwarded to matplotlib.

    Returns
    -------
    thetaMax : Maximum value of theta shown on the x-axis, as
            supplied by the user or computed by the function.

    h : array of Line2D handles of the individual plot lines.
    """

    data = np.asarray(data, dtype=np.double)
    m, n = data.shape  # `m` problems, `n` solvers

    # Check input
    if len(linestyle) < n:
        raise ValueError("Number of line specs < number of solvers")

    # Row-wise minima. NaN values are treated like +infinity.
    minVals = np.min(data, axis=1, initial=np.inf, where=~np.isnan(data))

    if np.any(minVals <= 0):
        raise ValueError("Data contains non-positive performance measurements")

    if thetaMax is None:
        thetaMax = _compute_theta_max(data, minVals)

    h: List[Any] = [None] * n
    for solver in range(n):  # for each solver
        col = _compute_theta(data[:, solver], minVals)  # performance ratio
        col = col[col <= thetaMax]  # crop and remove infs/NaNs

        if len(col) == 0:
            continue

        th, prob = _make_staircase(col, m, thetaMax, tol)

        # plot current line and disable frame clipping (to support y-intercept marking)
        h[solver] = plt.step(
            th,
            prob,
            linestyle[solver],
            color=colors[solver],
            where="post",
            **kwargs,
            zorder=n + 1 - solver,
        )
        h[solver][0].set_clip_on(False)

    # set axis limits
    plt.xlim([1, thetaMax])
    plt.ylim([0, 1.01])
    plt.xlabel(r"performance ratio")
    plt.ylabel(r"Proportion of problems")

    return thetaMax, h


def draw_pp(
    alg_names: list[str],
    callsM: np.ndarray,
    color_palette: dict[str, str],
    line_styles: dict[str, str],
    output_path: str | Path | None = None,
    fig_size: tuple[float, float] = (7, 5),
    title: str | None = None,
) -> None:
    """
    Draw and save performance profile plot.

    Parameters
    ----------
    alg_names : List of algorithm names
    callsM : (n_algorithms, n_problems) array of function calls
    color_palette : Dictionary mapping algorithm names to colors
    line_styles : Dictionary mapping algorithm names to line styles
    output_path : Path to save the figure (optional). If None, figure is not saved.
    fig_size : Figure size as (width, height) tuple. Defaults to (7, 5).
    title : Title for the plot (optional)
    """
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
