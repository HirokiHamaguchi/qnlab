from collections import deque
from typing import Deque, Tuple, Union

import numpy as np
import numpy.typing as npt

from qnlab.parameter import LineParameter
from qnlab.problem.base import BaseProblem
from qnlab.update.update import get_direction
from qnlab.util.callback import Callback
from qnlab.util.check_termination import check_termination
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


def qn_line(
    prob: BaseProblem,
    param: LineParameter,
    method: Method,
    callback: Union[Callback, None] = None,
) -> Tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    """Runs the L-BFGS algorithm for unconstrained optimization."""
    assert method.base == "Line"
    prob.reset()

    x = np.array(prob.x0, dtype=np.float64)

    if callback:
        callback.start(prob, x)

    fx = prob.f(x)
    g = prob.g(x)

    w = np.zeros(prob.n, dtype=np.float64)
    lm = QuasiNewtonMemory(g, param.m, method)
    pf: Deque[np.float64] = deque([], maxlen=param.past)

    d = -np.copy(g)

    step = np.float64(1.0) / np.linalg.norm(d)
    k = 0

    result = check_termination(g, param, fx, pf, k, prob.call_f)
    if result is not None:
        return result, fx, x

    while True:
        xp, fxp, gp = np.copy(x), fx, np.copy(g)

        ls, fx, step, x, g = param.linesearch(
            prob.n, x, fx, g, d, step, xp, gp, w, prob, param
        )
        k += 1

        if ls.is_error():
            if callback:
                callback.callback(prob, x, fx, g)
            x, g = np.copy(xp), np.copy(gp)
            return ls, fx, x

        if callback:
            callback.callback(prob, x, fx, g)

        result = check_termination(g, param, fx, pf, k, prob.call_f)
        if result is not None:
            return result, fx, x

        lm.add_new_data(x, fx, g, xp, fxp, gp, callback, prob.get_eps())
        d = get_direction(method, x, fx, g, lm)

        step = np.float64(1.0)
