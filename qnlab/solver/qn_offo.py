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
    """Run a first-order objective-function-free baseline."""
    gtol = np.float64(options.get("gtol", 1e-5))
    max_iterations = int(options.get("max_iterations", 15_000))
    max_evaluations = int(options.get("max_evaluations", 30_000))
    squared_offset = np.float64(options.get("offo_squared_offset", 1e-20))
    theta = np.float64(options.get("theta", 1.0))
    if (
        gtol < 0.0
        or max_iterations <= 0
        or max_evaluations <= 0
        or squared_offset <= 0.0
        or theta <= 0.0
    ):
        raise ValueError("Invalid OFFO parameter.")

    prob.reset()
    x = np.array(prob.x0, dtype=np.float64)
    if callback is not None:
        callback.start(prob, x)
    g = prob.g(x)
    fx = np.float64(np.nan)
    squared_scale = squared_offset

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

        squared_scale += np.dot(g, g)
        x = x - g / (theta * np.sqrt(squared_scale))
        g = prob.g(x)
        if callback is not None:
            callback.callback(prob, x, fx, g)

    raise RuntimeError("Unreachable OFFO termination state.")
