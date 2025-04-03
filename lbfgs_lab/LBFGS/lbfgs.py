from typing import Union

import numpy as np

from ._callback import CallbackData, IterationData
from ._lineSearch import owlqn_pseudo_gradient
from ._objectiveFunction import *
from ._params import LBFGSParameter, checkParams
from ._retValues import *


def lbfgs(
    n: int,
    x: npt.NDArray[np.float64],
    instance: ObjectiveFunction,
    param: Union[LBFGSParameter, None],
) -> Tuple[int, float, npt.NDArray[np.float64]]:
    if param is None:
        param = LBFGSParameter()

    m = param.m
    cd = CallbackData(n, instance)
    checkParams(n, param)

    # Allocate working arrays:
    x = np.array(x, dtype=float)
    xp = np.zeros(n, dtype=float)
    g = np.zeros(n, dtype=float)
    gp = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)
    # pseudo-gradient (for OWL-QN)
    pg = np.zeros(0 if param.orthantwise_c == 0.0 else n, dtype=float)
    # Allocate limited memory
    lm = [IterationData(0) for _ in range(m)]
    pf = np.zeros(param.past, dtype=float) if param.past > 0 else None

    # Evaluate the function and gradient at the initial point.
    fx, g = instance.evaluate(x, g, n, 0.0)
    if param.orthantwise_c != 0.0:
        xnorm = np.linalg.norm(
            x[param.orthantwise_start : param.orthantwise_end],
            ord=1,
        )
        fx += float(xnorm * param.orthantwise_c)
        pg = owlqn_pseudo_gradient(x, g, n, param)

    if pf is not None:
        pf[0] = fx

    d = np.copy(g if param.orthantwise_c == 0.0 else pg)

    xnorm = np.linalg.norm(x)
    if xnorm < 1.0:
        xnorm = 1.0
    gnorm = np.linalg.norm(g if param.orthantwise_c == 0.0 else pg)
    if gnorm / xnorm <= param.epsilon:
        return LBFGS_ALREADY_MINIMIZED, fx, x

    # Compute initial step: step = 1 / ||d||
    step = float(1.0 / np.linalg.norm(d))
    k = 1
    end = 0

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

        if ls < 0:
            x[:] = xp
            g[:] = gp
            return ls, fx, x

        xnorm = float(max(1.0, np.linalg.norm(x)))
        gnorm = float(np.linalg.norm(g if param.orthantwise_c == 0.0 else pg))

        ret = instance.progress(x, g, fx, xnorm, gnorm, step, n, k, ls)
        if ret:
            return ret, fx, x

        # Convergence test.
        if gnorm / xnorm <= param.epsilon:
            return LBFGS_SUCCESS, fx, x

        # Test stopping criterion.
        if pf is not None:
            if k >= param.past:
                rate = (pf[k % param.past] - fx) / fx
                if abs(rate) < param.delta:
                    return LBFGS_STOP, fx, x
            pf[k % param.past] = fx

        if param.max_iterations != 0 and k + 1 > param.max_iterations:
            return LBFGSERR_MAXIMUMITERATION, fx, x

        lm[end].s = x - xp
        lm[end].y = g - gp
        lm[end].ys = np.dot(lm[end].y, lm[end].s)
        yy = np.dot(lm[end].y, lm[end].y)
        if yy == 0.0:
            raise ValueError("? yy is 0.0")

        # Recursive formula to compute dir = -(H \cdot g).
        # This is described in page 779 of:
        # Jorge Nocedal.
        # Updating Quasi-Newton Matrices with Limited Storage.
        # Mathematics of Computation, Vol. 35, No. 151,
        # pp. 773--782, 1980.

        bound = m if k > m else k
        k += 1
        end = (end + 1) % m

        # --- Compute the new search direction using two-loop recursion ---
        d[:] = g if param.orthantwise_c == 0.0 else pg
        j = end
        for _ in range(bound):
            j = (j + m - 1) % m
            lm[j].alpha = np.dot(lm[j].s, d) / lm[j].ys
            d -= lm[j].alpha * lm[j].y
        scale = lm[end].ys / yy
        d *= scale
        for _ in range(bound):
            beta = np.dot(lm[j].y, d) / lm[j].ys
            d += (lm[j].alpha - beta) * lm[j].s
            j = (j + 1) % m

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
