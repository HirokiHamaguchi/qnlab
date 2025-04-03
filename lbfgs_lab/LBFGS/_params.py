from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

from ._callback import CallbackData
from ._retValues import *


class LBFGSParameter:
    """
    Class representing L-BFGS optimization parameters.
    This class mirrors the lbfgs_parameter_t struct from lbfgs.h.
    """

    def __init__(self):
        from ._lineSearch import LBFGS_LINESEARCH_DEFAULT, line_search_morethuente

        # The number of corrections to approximate the inverse Hessian.
        self.m: int = 6

        # Epsilon for the convergence test.
        self.epsilon: float = 1e-5

        # Distance for delta-based convergence test.
        self.past: int = 0

        # Delta for convergence test.
        self.delta: float = 1e-5

        # Maximum number of iterations (0 means continue until convergence or error).
        self.max_iterations: int = 0

        # Line search algorithm to use.
        self.linesearch_kind: int = LBFGS_LINESEARCH_DEFAULT
        self.linesearch: Callable[
            [
                int,
                npt.NDArray[np.float64],
                float,
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                float,
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                CallbackData,
                LBFGSParameter,
            ],
            Tuple[int, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]],
        ] = line_search_morethuente

        # Maximum number of trials for the line search.
        self.max_linesearch: int = 40

        # The minimum step size for the line search.
        self.min_step: float = 1e-20

        # The maximum step size for the line search.
        self.max_step: float = 1e20

        # Parameter to control the accuracy of the line search (sufficient decrease condition).
        self.ftol: float = 1e-4

        # Coefficient for the Wolfe condition.
        self.wolfe: float = 0.9

        # Additional accuracy parameter for the line search.
        self.gtol: float = 0.9

        # Machine precision parameter.
        self.xtol: float = 1e-16

        # Coefficient for the L1 norm (for OWL-QN method).
        # QWL-QN: Minimize F(x) + C |x|
        # Set 0 for standard minimization.
        self.orthantwise_c: float = 0.0

        # Start index for computing the L1 norm.
        self.orthantwise_start: int = 0

        # End index for computing the L1 norm.
        self.orthantwise_end: int = -1  # -1 will be converted to n


def checkParams(n: int, param: LBFGSParameter) -> None:
    error_code = _internal_check_params(n, param)
    if error_code != 0:
        raise ValueError(lbfgs_strerror(error_code))


def _internal_check_params(n: int, param: LBFGSParameter) -> int:
    from ._lineSearch import (
        LBFGS_LINESEARCH_BACKTRACKING,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
        LBFGS_LINESEARCH_MORETHUENTE,
        line_search_backtracking,
        line_search_backtracking_owlqn,
        line_search_morethuente,
    )

    if param.epsilon < 0.0:
        return LBFGSERR_INVALID_EPSILON
    if param.past < 0:
        return LBFGSERR_INVALID_TESTPERIOD
    if param.delta < 0.0:
        return LBFGSERR_INVALID_DELTA
    if param.min_step < 0.0:
        return LBFGSERR_INVALID_MINSTEP
    if param.max_step < param.min_step:
        return LBFGSERR_INVALID_MAXSTEP
    if param.ftol < 0.0:
        return LBFGSERR_INVALID_FTOL
    if (
        param.linesearch_kind == LBFGS_LINESEARCH_BACKTRACKING_WOLFE
        or param.linesearch_kind == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
    ):
        if param.wolfe <= param.ftol or 1.0 <= param.wolfe:
            return LBFGSERR_INVALID_WOLFE
    if param.gtol < 0.0:
        return LBFGSERR_INVALID_GTOL
    if param.xtol < 0.0:
        return LBFGSERR_INVALID_XTOL
    if param.max_linesearch <= 0:
        return LBFGSERR_INVALID_MAXLINESEARCH
    if param.orthantwise_c < 0.0:
        return LBFGSERR_INVALID_ORTHANTWISE
    if param.orthantwise_start < 0 or n < param.orthantwise_start:
        return LBFGSERR_INVALID_ORTHANTWISE_START

    if param.orthantwise_end < 0:
        param.orthantwise_end = n
    if n < param.orthantwise_end:
        return LBFGSERR_INVALID_ORTHANTWISE_END
    if param.orthantwise_c != 0.0:
        if param.linesearch_kind != LBFGS_LINESEARCH_BACKTRACKING:
            return LBFGSERR_INVALID_LINESEARCH
        else:
            param.linesearch = line_search_backtracking_owlqn
    else:
        if param.linesearch_kind == LBFGS_LINESEARCH_MORETHUENTE:
            param.linesearch = line_search_morethuente
        elif param.linesearch_kind in [
            LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
            LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
            LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE,
        ]:
            param.linesearch = line_search_backtracking
        else:
            return LBFGSERR_INVALID_LINESEARCH

    return 0
