from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

from ._callback import CallbackData
from ._retValues import RetCode


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
            Tuple[
                RetCode, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]
            ],
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

    def __str__(self):
        return (
            f"LBFGSParameter(m={self.m}, epsilon={self.epsilon}, past={self.past}, "
            f"delta={self.delta}, max_iterations={self.max_iterations}, "
            f"linesearch_kind={self.linesearch_kind}, max_linesearch={self.max_linesearch}, "
            f"min_step={self.min_step}, max_step={self.max_step}, ftol={self.ftol}, "
            f"wolfe={self.wolfe}, gtol={self.gtol}, xtol={self.xtol}, "
            f"orthantwise_c={self.orthantwise_c}, orthantwise_start={self.orthantwise_start}, "
            f"orthantwise_end={self.orthantwise_end})"
        )

    def checkParams(self, n: int) -> None:
        error_code = self._internal_check_params(n)
        if error_code != RetCode.SUCCESS:
            raise ValueError(str(error_code))

    def _internal_check_params(self, n: int) -> RetCode:
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

        if (self.m <= 0 and self.m != -1) or 1e10 <= self.m:
            return RetCode.ERR_INVALID_M
        if self.epsilon < 0.0:
            return RetCode.ERR_INVALID_EPSILON
        if self.past < 0:
            return RetCode.ERR_INVALID_TESTPERIOD
        if self.delta < 0.0:
            return RetCode.ERR_INVALID_DELTA
        if self.min_step < 0.0:
            return RetCode.ERR_INVALID_MINSTEP
        if self.max_step < self.min_step:
            return RetCode.ERR_INVALID_MAXSTEP
        if self.ftol < 0.0:
            return RetCode.ERR_INVALID_FTOL
        if (
            self.linesearch_kind == LBFGS_LINESEARCH_BACKTRACKING_WOLFE
            or self.linesearch_kind == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
        ):
            if self.wolfe <= self.ftol or 1.0 <= self.wolfe:
                return RetCode.ERR_INVALID_WOLFE
        if self.gtol < 0.0:
            return RetCode.ERR_INVALID_GTOL
        if self.xtol < 0.0:
            return RetCode.ERR_INVALID_XTOL
        if self.max_linesearch <= 0:
            return RetCode.ERR_INVALID_MAXLINESEARCH
        if self.orthantwise_c < 0.0:
            return RetCode.ERR_INVALID_ORTHANTWISE
        if self.orthantwise_start < 0 or n < self.orthantwise_start:
            return RetCode.ERR_INVALID_ORTHANTWISE_START

        if self.orthantwise_end < 0:
            self.orthantwise_end = n
        if n < self.orthantwise_end:
            return RetCode.ERR_INVALID_ORTHANTWISE_END
        if self.orthantwise_c != 0.0:
            if self.linesearch_kind != LBFGS_LINESEARCH_BACKTRACKING:
                return RetCode.ERR_INVALID_LINESEARCH
            else:
                self.linesearch = line_search_backtracking_owlqn
        else:
            if self.linesearch_kind == LBFGS_LINESEARCH_MORETHUENTE:
                self.linesearch = line_search_morethuente
            elif self.linesearch_kind in [
                LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
                LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
                LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE,
            ]:
                self.linesearch = line_search_backtracking
            else:
                return RetCode.ERR_INVALID_LINESEARCH

        return RetCode.SUCCESS
