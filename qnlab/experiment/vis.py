import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from qnlab.problem.base import BaseProblem
from qnlab.util.callback import Callback

# Constants
DEFAULT_LEVELS = 20
DEFAULT_MARKER_COUNT = 12
LINE_STYLES = ["-", "--", "-.", ":"]
GRADIENT_THRESHOLD = 1e-5


def _has_trajectory_data(prob: BaseProblem, callbacks: list[Callback]) -> bool:
    """Return whether every callback contains plottable optimization iterates."""
    return bool(callbacks) and all(
        len(callback.xs) > 0
        and all(np.asarray(x).shape == (prob.n,) for x in callback.xs)
        for callback in callbacks
    )


def _warn_missing_trajectory_data() -> None:
    warnings.warn(
        "Skipping the contour plot because optimization iterates (xs) are not "
        "available. Stored CUTEst results contain only calls, function values, "
        "and gradient norms.",
        RuntimeWarning,
        stacklevel=3,
    )


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
    prob: BaseProblem, callbacks: list[Callback], labels: list[str], levels: int, ax
) -> None:
    """Plot 2D contour with trajectories."""
    # Collect all points for bounds calculation
    all_points: list[list[float]] = []
    for callback in callbacks:
        x = np.array([prob.x0] + callback.xs)
        all_points.extend(x[:, :2].tolist())

    all_points_array = np.array(all_points)
    bounds = _get_plot_bounds(all_points_array)

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
    prob: BaseProblem, callbacks: list[Callback], labels: list[str], ax
) -> None:
    """Plot 1D function with trajectories."""
    all_xs: list[float] = []

    # Plot trajectories
    for i, callback in enumerate(callbacks):
        xs = [prob.x0] + callback.xs
        zs = [prob._f(x_k) for x_k in xs]
        is_lbfgs = "L-BFGS-B" in labels[i]
        color = "black" if is_lbfgs else None
        ax.plot(xs, zs, "o-", label=labels[i], alpha=0.5, color=color)
        all_xs.extend(float(np.asarray(x_k).item()) for x_k in xs)

    # Plot function
    min_x, max_x = min(all_xs), max(all_xs)
    X = np.linspace(min_x - 0.1, max_x + 0.1, 1000)
    Z = np.array([prob._f(x) for x in X])
    ax.plot(X, Z, "k--", label="$f(x)$", linewidth=2)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")


def _create_contour_plot(
    prob: BaseProblem,
    callbacks: list[Callback],
    labels: list[str],
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


def _get_marker_indices(
    x_values, target_count: int = DEFAULT_MARKER_COUNT
) -> list[int]:
    """Choose marker indices that are approximately uniform along the x-axis."""
    if target_count < 2:
        raise ValueError("target_count must be at least 2")

    x = np.asarray(x_values)
    if len(x) <= target_count:
        return list(range(len(x)))

    if np.all(np.diff(x) >= 0) and x[-1] > x[0]:
        targets = np.linspace(x[0], x[-1], target_count)
        indices = np.searchsorted(x, targets)
    else:
        indices = np.linspace(0, len(x) - 1, target_count, dtype=int)

    indices = np.clip(indices, 0, len(x) - 1)
    indices[0] = 0
    indices[-1] = len(x) - 1
    return np.unique(indices).tolist()


def _get_plot_properties(
    label: str,
    i: int,
    color_palette: dict | None = None,
    line_styles: dict[str, str] | None = None,
) -> dict:
    """Get plotting properties for a method."""
    return {
        "linewidth": 2.2,
        "alpha": 1.0,
        "color": (color_palette or {}).get(label, "black"),
        "fmt": (line_styles or {}).get(label, "o-"),
        "markersize": 6,
        "zorder": 5 - i / 10 if "SciPy" not in label else 10,
    }


def _plot_function_values(
    prob: BaseProblem,
    callbacks: list[Callback],
    labels: list[str],
    name: str,
    shift_val: float,
    x_axis: str,
    ax,
    color_palette: dict | None = None,
    line_styles: dict[str, str] | None = None,
) -> None:
    """Plot function values on the given axis."""
    xlabel = "oracle calls"  # default

    for i, callback in enumerate(callbacks):
        props = _get_plot_properties(
            labels[i], i, color_palette=color_palette, line_styles=line_styles
        )
        fmt = props.pop("fmt", None)
        shifted_fxs = [fx + shift_val for fx in callback.fxs]

        x_values: Sequence[int | float]
        if x_axis == "iterations":
            x_values = list(range(len(callback.fxs)))
            xlabel = "iterations"
        elif x_axis == "time":
            x_values = callback.times
            xlabel = "time (s)"
        else:
            x_values = callback.calls

        if fmt is None:
            ax.plot(x_values, shifted_fxs, label=labels[i], **props)
        else:
            props["markevery"] = _get_marker_indices(x_values)
            ax.plot(x_values, shifted_fxs, fmt, label=labels[i], **props)

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
    callbacks: list[Callback],
    labels: list[str],
    name: str,
    x_axis: str,
    ax,
    color_palette: dict | None = None,
    line_styles: dict[str, str] | None = None,
) -> None:
    """Plot gradient norms on the given axis."""
    xlabel = "oracle calls"  # default

    for i, callback in enumerate(callbacks):
        # Special handling for L-BFGS-B variants
        is_lbfgs = "L-BFGS-B" in labels[i] and "(m=" not in labels[i]
        props = _get_plot_properties(
            labels[i], i, color_palette=color_palette, line_styles=line_styles
        )
        fmt = props.pop("fmt", None)
        if fmt is None:
            props["linestyle"] = _get_line_style(i, is_lbfgs)

        x_values: Sequence[int | float]
        if x_axis == "iterations":
            x_values = list(range(len(callback.gnorms)))
            xlabel = "iterations"
        elif x_axis == "time":
            x_values = callback.times
            xlabel = "time (s)"
        else:
            x_values = callback.calls

        if fmt is None:
            ax.plot(x_values, callback.gnorms, label=labels[i], **props)
        else:
            props["markevery"] = _get_marker_indices(x_values)
            ax.plot(x_values, callback.gnorms, fmt, label=labels[i], **props)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$||\\nabla f(x)||$")

    if name:
        ax.set_title(name + f" ($n={prob.n}$)")

    ax.axhline(y=GRADIENT_THRESHOLD, color="black", linestyle="--", linewidth=2)
    ax.legend()


def _calculate_shift_value(callbacks: list[Callback]) -> float:
    """Calculate shift value for function values to ensure positivity."""
    min_fx = min(min(float(fx) for fx in callback.fxs) for callback in callbacks)
    shift_val = max(0, -min_fx)
    return shift_val + 1e-5 if shift_val > 0 else 0


def _truncate_callbacks(
    callbacks: list[Callback], max_length: int, x_axis: str = "calls"
) -> None:
    """Truncate callback data to maximum length."""
    for callback in callbacks:
        if x_axis == "iterations":
            idx = min(max_length, len(callback.calls))
        elif x_axis == "time":
            idx = 0
            for i, elapsed in enumerate(callback.times):
                if elapsed > max_length:
                    idx = i
                    break
            else:
                idx = len(callback.times)
        else:
            idx = 0
            for i, call in enumerate(callback.calls):
                if call > max_length:
                    idx = i
                    break
            else:
                idx = len(callback.calls)

        callback.xs = callback.xs[:idx]
        callback.fxs = callback.fxs[:idx]
        callback.gnorms = callback.gnorms[:idx]
        callback.calls = callback.calls[:idx]
        callback.times = callback.times[:idx]


def _save_or_show_figure(pdf_path: str, suffix: str = "") -> None:
    """Save figure to PDF or show it."""
    plt.tight_layout()
    if pdf_path:
        assert not pdf_path.endswith(".pdf"), (
            "PDF path should not include .pdf extension"
        )
        plt.savefig(f"{pdf_path}{suffix}.pdf", bbox_inches="tight")
    else:
        plt.show()


def vis(
    prob: BaseProblem,
    callbacks: list[Callback],
    labels: list[str],
    name: str,
    only_grad: bool = False,
    only_plot: bool = False,
    levels: int = DEFAULT_LEVELS,
    pdf_path: str = "",
    max_length: int = int(1e8),
    one_figure: bool = False,
    use_tex: bool = False,
    x_axis: Literal["calls", "iterations", "time"] = "calls",
    color_palette: dict | None = None,
    line_styles: dict[str, str] | None = None,
) -> None:
    assert x_axis in ["calls", "iterations", "time"]

    _configure_matplotlib(use_tex)
    _truncate_callbacks(callbacks, max_length, x_axis)

    if one_figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        if _has_trajectory_data(prob, callbacks):
            _create_contour_plot(prob, callbacks, labels, "", DEFAULT_LEVELS, ax1)
        else:
            _warn_missing_trajectory_data()
            ax1.set_axis_off()
            ax1.text(
                0.5,
                0.5,
                "Trajectory data not available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
        _plot_gradient_norms(
            prob,
            callbacks,
            labels,
            "",
            x_axis,
            ax2,
            color_palette,
            line_styles,
        )
        fig.suptitle(name, y=0.95)
    else:
        if not only_plot:
            if _has_trajectory_data(prob, callbacks):
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                _create_contour_plot(prob, callbacks, labels, name, levels, ax)
                _save_or_show_figure(pdf_path, "_contour")
            else:
                _warn_missing_trajectory_data()

        shift_val = _calculate_shift_value(callbacks)

        if only_grad:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            _plot_gradient_norms(
                prob,
                callbacks,
                labels,
                name,
                x_axis,
                ax,
                color_palette,
                line_styles,
            )
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            _plot_function_values(
                prob,
                callbacks,
                labels,
                name,
                shift_val,
                x_axis,
                ax1,
                color_palette,
                line_styles,
            )
            _plot_gradient_norms(
                prob,
                callbacks,
                labels,
                name,
                x_axis,
                ax2,
                color_palette,
                line_styles,
            )

    _save_or_show_figure(pdf_path)
    plt.rcdefaults()
    plt.close()
