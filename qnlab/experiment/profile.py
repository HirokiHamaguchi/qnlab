"""
The code is based on `perfprof` from the MATLAB Guide
by D. J. Higham and N. J. Higham:
https://github.com/higham/matlab-guide-3ed/blob/master/perfprof.m
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TOL = np.sqrt(np.finfo(np.double).eps)


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
    tol=DEFAULT_TOL,
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

    h: list[Any] = [None] * n
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


def data_profile(
    data,
    dimensions,
    linestyle,
    colors,
    alpha_max=None,
    **kwargs,
) -> Any:
    """Plot fractions solved within ``alpha * (problem dimension + 1)`` calls."""
    data = np.asarray(data, dtype=np.double)
    dimensions = np.asarray(dimensions, dtype=np.double)
    m, n = data.shape
    if dimensions.shape != (m,):
        raise ValueError("One problem dimension is required for each data row")
    if np.any(dimensions <= 0):
        raise ValueError("Problem dimensions must be positive")
    if len(linestyle) < n:
        raise ValueError("Number of line specs < number of solvers")

    normalized = data / (dimensions[:, None] + 1.0)
    finite = normalized[np.isfinite(normalized)]
    if alpha_max is None:
        alpha_max = max(float(np.max(finite, initial=1.0)), 1.0)

    handles: list[Any] = [None] * n
    for solver in range(n):
        solved = np.sort(normalized[np.isfinite(normalized[:, solver]), solver])
        solved = solved[solved <= alpha_max]
        if len(solved) == 0:
            continue
        alpha = np.insert(solved, 0, 0.0)
        proportion = np.arange(len(solved) + 1) / m
        handles[solver] = plt.step(
            alpha,
            proportion,
            linestyle[solver],
            color=colors[solver],
            where="post",
            **kwargs,
            zorder=n + 1 - solver,
        )

    plt.xlim([0, alpha_max])
    plt.ylim([0, 1.01])
    plt.xlabel(r"normalized oracle-call budget $\alpha$")
    plt.ylabel(r"Proportion of problems solved")
    return alpha_max, handles
