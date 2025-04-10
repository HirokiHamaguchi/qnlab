from collections import deque
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from .util.callback import CallbackData
from .util.lineSearch import owlqn_pseudo_gradient
from .util.objectiveFunction import ObjectiveFunction
from .util.params import LBFGSParameter
from .util.retValues import RetCode
from .util.utils import _check_termination, _two_loop_recursion, _update_lm


def lbfgs(
    instance: ObjectiveFunction,
    param: Union[LBFGSParameter, None] = None,
) -> Tuple[RetCode, float, npt.NDArray[np.float64]]:
    """Runs the L-BFGS algorithm for unconstrained optimization.

    Args:
        instance (ObjectiveFunction): The objective function to be minimized.
        param (LBFGSParameter | None): The L-BFGS parameters. If None, default parameters are used.

    Returns:
        Tuple[RetCode, float, numpy.ndarray]: A tuple of (result code, final function value, final point).
    """
    n, x = instance.n, instance.x0
    if param is None:
        param = LBFGSParameter()
    assert isinstance(param, LBFGSParameter), "param must be of type LBFGSParameter"
    param.checkParams(n)

    m = param.m
    cd = CallbackData(n, instance)

    # Allocate working arrays:
    x = np.array(x, dtype=float)
    xp = np.zeros(n, dtype=float)
    g = np.zeros(n, dtype=float)
    gp = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)
    # pseudo-gradient (for OWL-QN)
    pg = np.zeros(0 if param.orthantwise_c == 0.0 else n, dtype=float)
    # Allocate limited memory
    lm = deque([], maxlen=m)
    pf = np.zeros(param.past, dtype=float) if param.past > 0 else None

    # Evaluate the function and gradient at the initial point.
    fx, g = instance.evaluate(x)
    if param.orthantwise_c != 0.0:
        xnorm = np.linalg.norm(
            x[param.orthantwise_start : param.orthantwise_end],
            ord=1,
        )
        fx += float(xnorm * param.orthantwise_c)
        pg = owlqn_pseudo_gradient(x, g, n, param)

    if pf is not None:
        pf[0] = fx

    d = -np.copy(g if param.orthantwise_c == 0.0 else pg)

    xnorm = max(1.0, np.linalg.norm(x))
    gnorm = np.linalg.norm(g if param.orthantwise_c == 0.0 else pg)
    if gnorm / xnorm <= param.epsilon:
        return RetCode.ALREADY_MINIMIZED, fx, x

    # Compute initial step: step = 1 / ||d||
    step = float(1.0 / np.linalg.norm(d))
    k = 1

    while True:
        # Save the current x and gradient
        xp[:] = x
        gp[:] = g

        # --- Line search ---
        if param.orthantwise_c == 0.0:
            ls, fx, step, x, g = param.linesearch(
                n, x, fx, g, d, step, xp, gp, w, cd, param
            )
        else:
            ls, fx, step, x, g = param.linesearch(
                n, x, fx, g, d, step, xp, pg, w, cd, param
            )
            pg = owlqn_pseudo_gradient(x, g, n, param)

        if ls.is_error():
            x[:] = xp
            g[:] = gp
            return ls, fx, x

        xnorm = float(max(1.0, np.linalg.norm(x)))
        gnorm = float(np.linalg.norm(g if param.orthantwise_c == 0.0 else pg))

        instance.progress(fx, gnorm, step, k)

        # Convergence test.
        result = _check_termination(gnorm / xnorm, param, fx, pf, k)
        if result is not None:
            return result, fx, x

        ys, yy = _update_lm(lm, x, g, xp, gp)
        k += 1
        if param.orthantwise_c == 0.0:
            d = -np.copy(g)
        else:
            d = -np.copy(pg)
        d = _two_loop_recursion(d, lm, ys, yy)

        # For OWL-QN, constrain the search direction.
        if param.orthantwise_c != 0.0:
            isInvalid = (
                d[param.orthantwise_start : param.orthantwise_end]
                * pg[param.orthantwise_start : param.orthantwise_end]
                >= 0
            )
            d[param.orthantwise_start : param.orthantwise_end][isInvalid] = 0.0

        # Reset step to 1 for the next iteration.
        step = 1.0
