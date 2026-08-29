from collections import deque

import numpy as np
import numpy.typing as npt

from qnlab.parameter import LineParameter, NTRQNParameter, OwlParameter
from qnlab.util.ret_values import RetCode


def check_termination(
    g: npt.NDArray[np.float64],
    param: LineParameter | OwlParameter | NTRQNParameter,
    fx: np.float64,
    pf: deque,
    k: int,
    eval_count: int,
) -> RetCode | None:
    """Check for termination based on gradient norm, function values, and
    iteration count.
    """
    if 0 < param.past <= len(pf) and abs(max(pf) - fx) < param.ftol * abs(fx):
        return RetCode.STOP
    pf.append(fx)
    if np.linalg.norm(g, ord=np.inf) <= param.gtol:
        return RetCode.SUCCESS if k > 0 else RetCode.ALREADY_MINIMIZED
    if eval_count >= param.max_evaluations:
        return RetCode.ERR_MAXIMUMEVALUATION
    if 0 < param.max_iterations <= k:
        return RetCode.ERR_MAXIMUMITERATION
    return None
