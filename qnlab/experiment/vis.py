from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qnlab.problem.base import BaseProblem
from qnlab.util.callback import Callback

# Constants
DEFAULT_LEVELS = 20
LINE_STYLES = ["-", "--", "-.", ":"]
GRADIENT_THRESHOLD = 1e-5


def _configure_matplotlib(use_tex: bool = False) -> None:
    """Configure matplotlib settings."""
    sns.set_style("darkgrid")

    if use_tex:
        plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

    # Configure font sizes
    plt.rcParams["axes.titlesize"] = 30
    plt.rcParams["figure.titlesize"] = 30
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 15


def _get_plot_bounds(xs: np.ndarray) -> tuple:
    """Calculate plot bounds with padding."""
    x_min, x_max = xs[:, 0].min(), xs[:, 0].max()
    y_min, y_max = xs[:, 1].min(), xs[:, 1].max()
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    return (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)


def _create_contour_grid(prob: BaseProblem, bounds: tuple) -> tuple:
    """Create contour grid for 2D problems."""
    x_min, x_max, y_min, y_max = bounds

    if prob.n <= 100:
        grid_size = 100
    else:
        grid_size = 10

    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )

    Z = np.zeros(X.shape)
    has_x_opt = hasattr(prob, "x_opt")

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if has_x_opt:
                point = np.array([X[i, j], Y[i, j]] + prob.x_opt[2:].tolist())
            else:
                point = np.array([X[i, j], Y[i, j]] + [0] * (prob.n - 2))
            Z[i, j] = prob._f(point)

    return X, Y, Z


def _plot_2d_contour(
    prob: BaseProblem, callbacks: List[Callback], labels: List[str], levels: int, ax
) -> None:
    """Plot 2D contour with trajectories."""
    # Collect all points for bounds calculation
    all_points = []
    for callback in callbacks:
        x = np.array([prob.x0] + callback.xs)
        all_points.extend(x[:, :2].tolist())

    all_points = np.array(all_points)
    bounds = _get_plot_bounds(all_points)

    # Plot trajectories
    for i, callback in enumerate(callbacks):
        x = np.array([prob.x0] + callback.xs)
        is_lbfgs = "L-BFGS-B" in labels[i]
        ax.plot(
            x[:, 0],
            x[:, 1],
            "o-",
            label=labels[i],
            alpha=0.5,
            color="black" if is_lbfgs else None,
        )

    # Create and plot contour
    X, Y, Z = _create_contour_grid(prob, bounds)
    ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def _plot_1d_function(
    prob: BaseProblem, callbacks: List[Callback], labels: List[str], ax
) -> None:
    """Plot 1D function with trajectories."""
    all_xs = []

    # Plot trajectories
    for i, callback in enumerate(callbacks):
        xs = [prob.x0] + callback.xs
        zs = [prob._f(x_k) for x_k in xs]
        is_lbfgs = "L-BFGS-B" in labels[i]
        color = "black" if is_lbfgs else None
        ax.plot(xs, zs, "o-", label=labels[i], alpha=0.5, color=color)
        all_xs.extend(xs)

    # Plot function
    min_x, max_x = min(all_xs), max(all_xs)
    X = np.linspace(min_x - 0.1, max_x + 0.1, 1000)
    Z = np.array([prob._f(x) for x in X])
    ax.plot(X, Z, "k--", label="$f(x)$", linewidth=2)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")


def _create_contour_plot(
    prob: BaseProblem,
    callbacks: List[Callback],
    labels: List[str],
    name: str,
    levels: int,
    ax,
):
    if prob.n >= 2:
        _plot_2d_contour(prob, callbacks, labels, levels, ax)
    else:
        _plot_1d_function(prob, callbacks, labels, ax)
    if name:
        ax.set_title(name)
    ax.legend()
    return ax


def _get_line_style(i: int, is_lbfgs: bool) -> str:
    """Get line style for plotting."""
    return "-" if is_lbfgs else LINE_STYLES[i % len(LINE_STYLES)]


def _get_plot_properties(label: str, i: int) -> dict:
    """Get plotting properties for a method."""
    is_lbfgs = "S_L-BFGS-B" in label
    return {
        "linewidth": 4,
        "alpha": 1.0 if is_lbfgs else 0.8,
        "color": "black" if is_lbfgs else None,
        "linestyle": _get_line_style(i, is_lbfgs),
    }


def _plot_function_values(
    prob: BaseProblem,
    callbacks: List[Callback],
    labels: List[str],
    name: str,
    shift_val: float,
    x_axis: str,
    ax,
) -> None:
    """Plot function values on the given axis."""
    xlabel = "oracle calls"  # default

    for i, callback in enumerate(callbacks):
        props = _get_plot_properties(labels[i], i)
        shifted_fxs = [fx + shift_val for fx in callback.fxs]

        if x_axis == "iterations":
            x_values = list(range(len(callback.fxs)))
            xlabel = "iterations"
        else:
            x_values = callback.calls

        ax.plot(x_values, shifted_fxs, label=labels[i], **props)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)

    ylabel = "$f(x)$"
    if shift_val > 0:
        ylabel += f" (+{shift_val:.2f})"
    ax.set_ylabel(ylabel)

    ax.set_title(name + f" ($n={prob.n}$)")
    ax.legend()


def _plot_gradient_norms(
    prob: BaseProblem,
    callbacks: List[Callback],
    labels: List[str],
    name: str,
    x_axis: str,
    ax,
) -> None:
    """Plot gradient norms on the given axis."""
    xlabel = "oracle calls"  # default

    for i, callback in enumerate(callbacks):
        # Special handling for L-BFGS-B variants
        is_lbfgs = "L-BFGS-B" in labels[i] and "(m=" not in labels[i]
        props = _get_plot_properties(labels[i], i)
        props["linestyle"] = _get_line_style(i, is_lbfgs)

        if x_axis == "iterations":
            x_values = list(range(len(callback.gnorms)))
            xlabel = "iterations"
        else:
            x_values = callback.calls

        ax.plot(x_values, callback.gnorms, label=labels[i], **props)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$||\\nabla f(x)||$")

    if name:
        ax.set_title(name + f" ($n={prob.n}$)")

    ax.axhline(y=GRADIENT_THRESHOLD, color="black", linestyle="--", linewidth=2)
    ax.legend()


def _calculate_shift_value(callbacks: List[Callback]) -> float:
    """Calculate shift value for function values to ensure positivity."""
    min_fx = min(min(callback.fxs) for callback in callbacks)
    shift_val = max(0, -min_fx)
    return shift_val + 1e-5 if shift_val > 0 else 0


def _truncate_callbacks(callbacks: List[Callback], max_length: int) -> None:
    """Truncate callback data to maximum length."""
    for callback in callbacks:
        callback.xs = callback.xs[:max_length]
        callback.fxs = callback.fxs[:max_length]
        callback.gnorms = callback.gnorms[:max_length]
        callback.calls = callback.calls[:max_length]


def _save_or_show_figure(pdf_path: str, suffix: str = "") -> None:
    """Save figure to PDF or show it."""
    plt.tight_layout()
    if pdf_path:
        assert not pdf_path.endswith(
            ".pdf"
        ), "PDF path should not include .pdf extension"
        plt.savefig(f"{pdf_path}{suffix}.pdf", bbox_inches="tight")
    else:
        plt.show()


def vis(
    prob: BaseProblem,
    callbacks: List[Callback],
    labels: List[str],
    name: str,
    only_grad: bool = False,
    only_plot: bool = False,
    levels: int = DEFAULT_LEVELS,
    pdf_path: str = "",
    max_length: int = int(1e8),
    one_figure: bool = False,
    use_tex: bool = False,
    x_axis: str = "calls",
) -> None:
    assert x_axis in ["calls", "iterations"]

    _configure_matplotlib(use_tex)
    _truncate_callbacks(callbacks, max_length)

    if one_figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        _create_contour_plot(prob, callbacks, labels, "", DEFAULT_LEVELS, ax1)
        _plot_gradient_norms(prob, callbacks, labels, "", x_axis, ax2)
        fig.suptitle(name, y=0.95)
    else:
        if not only_plot:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            _create_contour_plot(prob, callbacks, labels, name, levels, ax)
            _save_or_show_figure(pdf_path, "_contour")

        shift_val = _calculate_shift_value(callbacks)

        if only_grad:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            _plot_gradient_norms(prob, callbacks, labels, name, x_axis, ax)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            _plot_function_values(prob, callbacks, labels, name, shift_val, x_axis, ax1)
            _plot_gradient_norms(prob, callbacks, labels, name, x_axis, ax2)

    _save_or_show_figure(pdf_path)
    plt.rcdefaults()
