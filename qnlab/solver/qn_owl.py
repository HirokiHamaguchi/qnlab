from collections import deque
from typing import Deque, Tuple, Union

import numpy as np
import numpy.typing as npt

from qnlab.parameter import OwlParameter
from qnlab.problem.base import BaseProblem
from qnlab.update.update import get_direction
from qnlab.util.callback import Callback
from qnlab.util.check_termination import check_termination
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method
from qnlab.util.owlqn_pseudo_gradient import owlqn_pseudo_gradient
from qnlab.util.ret_values import RetCode


def qn_owl(
    prob: BaseProblem,
    param: OwlParameter,
    method: Method = Method(store="raw", secant="raw", update="bfgs"),
    callback: Union[Callback, None] = None,
) -> Tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    """Runs the OWL-QN algorithm for optimization with orthant-wise constraints."""
    prob.reset()

    # Allocate working arrays:
    x = np.array(prob.x0, dtype=np.float64)

    if callback:
        callback.start(prob, x)

    # Evaluate the function and gradient at the initial point.
    fx = prob.f(x)
    g = prob.g(x)

    w = np.zeros(prob.n, dtype=np.float64)
    lm = QuasiNewtonMemory(g, param.m, method)
    pf: Deque[np.float64] = deque([], maxlen=param.past)

    xnorm = np.linalg.norm(
        x[param.orthantwise_start : param.orthantwise_end],
        ord=1,
    )
    fx += xnorm * param.orthantwise_c
    pg = owlqn_pseudo_gradient(x, g, prob.n, param)

    d = -np.copy(pg)

    # Compute initial step: step = 1 / ||d||
    step = np.float64(1.0) / np.linalg.norm(d)
    k = 1

    result = check_termination(pg, param, fx, pf, k, prob.call_f)
    if result is not None:
        return result, fx, x

    while True:
        # Save the current x and gradient
        xp, fxp, gp = np.copy(x), fx, np.copy(g)

        # --- Line search ---
        ls, fx, step, x, g = param.linesearch(
            prob.n, x, fx, g, d, step, xp, pg, w, prob, param
        )
        pg = owlqn_pseudo_gradient(x, g, prob.n, param)

        if ls.is_error():
            x, g = np.copy(xp), np.copy(gp)
            return ls, fx, x

        if callback:
            callback.callback(prob, x, fx, pg)

        # Convergence test.
        result = check_termination(pg, param, fx, pf, k, prob.call_f)
        if result is not None:
            return result, fx, x

        lm.add_new_data(x, fx, g, xp, fxp, gp, callback, prob.get_eps())
        k += 1
        d = get_direction(method, x, fx, pg, lm)

        # For OWL-QN, constrain the search direction.
        isInvalid = (
            d[param.orthantwise_start : param.orthantwise_end]
            * pg[param.orthantwise_start : param.orthantwise_end]
            >= 0
        )
        d[param.orthantwise_start : param.orthantwise_end][isInvalid] = 0.0

        # Reset step to 1 for the next iteration.
        step = np.float64(1.0)
