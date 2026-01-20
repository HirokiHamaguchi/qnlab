from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from qnlab.parameter.param import BaseParameter
from qnlab.problem.base import BaseProblem
from qnlab.util.ret_values import RetCode


class LineParameter(BaseParameter):
    """Parameters for line-search-based quasi-Newton solvers."""

    def __init__(
        self, n: int, options: Optional[Dict[str, Union[np.float64, int]]] = None
    ) -> None:
        super().__init__(n, options=None)

        self.ftol: np.float64 = np.float64(0.0)
        self.past: int = 10

        from qnlab.util.linesearch import LINESEARCH_DEFAULT, line_search_morethuente

        self.linesearch_kind: int = LINESEARCH_DEFAULT
        self.linesearch: Callable[
            [
                int,
                npt.NDArray[np.float64],
                np.float64,
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                np.float64,
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                BaseProblem,
                Any,
            ],
            Tuple[
                RetCode,
                np.float64,
                np.float64,
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
            ],
        ] = line_search_morethuente
        self.max_linesearch: int = 40
        self.min_step: np.float64 = np.float64(1e-20)
        self.max_step: np.float64 = np.float64(1e20)
        self.armijo: np.float64 = np.float64(1e-4)
        self.wolfe: np.float64 = np.float64(0.9)
        self.xtol: np.float64 = np.float64(1e-16)

        if options:
            self._apply_options(options)

        self._validate()

    def _validate(self) -> None:
        code = self._check_shared()
        if code.is_error():
            raise ValueError(str(code))
        code = self._check_line_params()
        if code.is_error():
            raise ValueError(str(code))

    def _check_line_params(self) -> RetCode:
        from qnlab.util.linesearch import (
            LINESEARCH_BACKTRACKING_ARMIJO,
            LINESEARCH_BACKTRACKING_STRONG_WOLFE,
            LINESEARCH_BACKTRACKING_WOLFE,
            LINESEARCH_MORETHUENTE,
            line_search_backtracking,
            line_search_morethuente,
        )

        if self.ftol < 0.0:
            return RetCode.ERR_INVALID_FTOL
        if self.past < 0:
            return RetCode.ERR_INVALID_TESTPERIOD
        if self.min_step < 0.0:
            return RetCode.ERR_INVALID_MINSTEP
        if self.max_step < self.min_step:
            return RetCode.ERR_INVALID_MAXSTEP
        if self.armijo < 0.0:
            return RetCode.ERR_INVALID_ARMIJO
        if self.linesearch_kind in (
            LINESEARCH_BACKTRACKING_WOLFE,
            LINESEARCH_BACKTRACKING_STRONG_WOLFE,
        ):
            if self.wolfe <= self.armijo or 1.0 <= self.wolfe:
                return RetCode.ERR_INVALID_WOLFE
        if self.xtol < 0.0:
            return RetCode.ERR_INVALID_XTOL
        if self.max_linesearch <= 0:
            return RetCode.ERR_INVALID_MAXLINESEARCH

        if self.linesearch_kind == LINESEARCH_MORETHUENTE:
            self.linesearch = line_search_morethuente
        elif self.linesearch_kind in (
            LINESEARCH_BACKTRACKING_ARMIJO,
            LINESEARCH_BACKTRACKING_WOLFE,
            LINESEARCH_BACKTRACKING_STRONG_WOLFE,
        ):
            self.linesearch = line_search_backtracking
        else:
            return RetCode.ERR_INVALID_LINESEARCH

        return RetCode.SUCCESS
