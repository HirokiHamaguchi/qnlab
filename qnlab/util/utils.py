from typing import Union, Tuple
from collections import deque

import numpy as np
import numpy.typing as npt

from .callback import IterationData
from .params import LBFGSParameter
from .retValues import RetCode


def _check_termination(
    gnorm_div_xnorm: float,
    param: LBFGSParameter,
    fx: float,
    pf: Union[npt.NDArray[np.float64], None],
    k: int,
) -> Union[RetCode, None]:
    """Check for termination based on gradient norm, function values, and
    iteration count.

    :param gnorm_div_xnorm: Ratio of gradient norm to x norm :param
    param: LBFGS parameters :param fx: Current function value :param pf:
    Array of past function values :param k: Current iteration index
    :return: A RetCode if a termination condition is met, otherwise None
    """
    if gnorm_div_xnorm <= param.epsilon:
        return RetCode.SUCCESS
    if pf is not None:
        if k >= param.past:
            rate = (pf[k % param.past] - fx) / fx
            if abs(rate) < param.delta:
                return RetCode.STOP
    if param.max_iterations != 0 and k + 1 > param.max_iterations:
        return RetCode.ERR_MAXIMUMITERATION
    return None


def _update_lm(
    lm: deque,
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64],
    gp: npt.NDArray[np.float64],
) -> Tuple[float, float]:
    """Update the limited-memory storage with iteration data.

    :param lm: Deque storing iteration data :param x: Current variable
    vector :param g: Current gradient :param xp: Previous variable
    vector :param gp: Previous gradient :return: Tuple of dot products
    (ys, yy)
    """
    new_data = IterationData(0)
    new_data.s = x - xp
    new_data.y = g - gp
    new_data.ys = np.dot(new_data.y, new_data.s)
    yy = np.dot(new_data.y, new_data.y)
    if yy == 0.0:
        raise ValueError(str(RetCode.ERR_LOGICERROR))
    lm.append(new_data)
    return new_data.ys, yy


def _two_loop_recursion(d, lm: deque, ys: float, yy: float) -> npt.NDArray[np.float64]:
    """Perform two-loop recursion to compute the search direction.

    :param d: Current direction (will be updated) :param lm: Deque of
    iteration data :param m: Number of stored corrections :param ys:
    Inner product of y and s :param yy: Inner product of y and y
    :return: Updated direction
    """
    # Recursive formula to compute dir = -(H \cdot g).
    # This is described in page 779 of:
    # Jorge Nocedal.
    # Updating Quasi-Newton Matrices with Limited Storage.
    # Mathematics of Computation, Vol. 35, No. 151,
    # pp. 773--782, 1980.
    for item in reversed(lm):
        item.alpha = np.dot(item.s, d) / item.ys
        d -= item.alpha * item.y
    scale = ys / yy
    d *= scale
    for item in lm:
        beta = np.dot(item.y, d) / item.ys
        d += (item.alpha - beta) * item.s
    return d
