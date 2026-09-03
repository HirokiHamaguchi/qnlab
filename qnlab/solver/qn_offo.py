import numpy as np
import numpy.typing as npt

from qnlab.problem.base import BaseProblem
from qnlab.util.callback import Callback
from qnlab.util.ret_values import RetCode


def qn_offo(
    prob: BaseProblem,
    options: dict[str, np.float64 | int],
    callback: Callback | None = None,
) -> tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    """Run the first-order ASTR1-Adagrad method of Gratton et al.

    This is the ``adagrad`` variant in Table 1 of Gratton, Jerad, and Toint,
    *Complexity of a Class of First-Order Objective-Function-Free Optimization
    Algorithms*, doi:10.1080/10556788.2023.2296431. It uses ``B_k = 0``,
    ``gamma_k = 1``, ``mu = 1/2``, and ``vartheta = theta = 1`` with the
    paper's default ``varsigma = 1/100``. The caller may retain a common
    stopping rule and iteration budget for benchmark comparability.
    """
    gtol = np.float64(options.get("gtol", 1e-5))
    max_iterations = int(options.get("max_iterations", 15_000))
    max_evaluations = int(options.get("max_evaluations", 30_000))
    varsigma = np.float64(options.get("varsigma", 1e-2))
    theta = np.float64(options.get("theta", 1.0))
    if (
        gtol < 0.0
        or max_iterations <= 0
        or max_evaluations <= 0
        or varsigma <= 0.0
        or theta <= 0.0
    ):
        raise ValueError("Invalid ASTR1-Adagrad parameter.")

    prob.reset()
    x = np.array(prob.x0, dtype=np.float64)
    if callback is not None:
        callback.start(prob, x)
    g = prob.g(x)
    fx = np.float64(np.nan)
    squared_weights = np.full(prob.n, varsigma, dtype=np.float64)

    for iteration in range(max_iterations + 1):
        if not np.all(np.isfinite(g)):
            return RetCode.ERR_NUMERICAL_OVERFLOW, fx, x
        if np.linalg.norm(g, ord=np.inf) <= gtol:
            return (
                RetCode.SUCCESS if iteration > 0 else RetCode.ALREADY_MINIMIZED,
                fx,
                x,
            )
        if iteration == max_iterations:
            return RetCode.ERR_MAXIMUMITERATION, fx, x
        if prob.count_calls() >= max_evaluations:
            return RetCode.ERR_MAXIMUMEVALUATION, fx, x

        squared_weights += np.square(g)
        x = x - g / (theta * np.sqrt(squared_weights))
        g = prob.g(x)
        if callback is not None:
            callback.callback(prob, x, fx, g)

    raise RuntimeError("Unreachable ASTR1-Adagrad termination state.")
