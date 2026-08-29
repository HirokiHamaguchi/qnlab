import numpy as np

from qnlab.parameter.param import BaseParameter
from qnlab.util.ret_values import RetCode


class NTRQNParameter(BaseParameter):
    """Parameters for NTRQN relaxed Armijo solver."""

    def __init__(
        self, n: int, options: dict[str, np.float64 | int] | None = None
    ) -> None:
        super().__init__(n, options=None)

        self.ftol: np.float64 = np.float64(0.0)
        self.past: int = 10
        self.max_linesearch: int = 40
        self.armijo: np.float64 = np.float64(1e-4)
        self.max_inf_nan_rejections: int = 10
        self.mu_scale: np.float64 = np.float64(0.1)
        self.mu_min_fraction: np.float64 = np.float64(0.01)
        self.non_monotone: int = 1
        self.offo_squared_offset: np.float64 = np.float64(1e-20)
        self.restart_threshold: np.float64 = np.float64(np.inf)
        self.max_restarts: int = 0
        self.force_offo: int = 0

        if options:
            self._apply_options(options)

        self._validate()

    def _validate(self) -> None:
        code = self._check_shared()
        if code.is_error():
            raise ValueError(str(code))
        code = self._check_ntrqn_params()
        if code.is_error():
            raise ValueError(str(code))

    def _check_ntrqn_params(self) -> RetCode:
        if self.ftol < 0.0:
            return RetCode.ERR_INVALID_FTOL
        if self.past < 0:
            return RetCode.ERR_INVALID_TESTPERIOD
        if self.max_linesearch <= 0:
            return RetCode.ERR_INVALID_MAXLINESEARCH
        if self.armijo < 0.0:
            return RetCode.ERR_INVALID_ARMIJO
        if self.max_inf_nan_rejections <= 0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.mu_scale <= 0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.mu_min_fraction <= 0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.non_monotone <= 0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.offo_squared_offset <= 0.0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.restart_threshold <= 0.0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.max_restarts < 0:
            return RetCode.ERR_INVALIDPARAMETERS
        if self.force_offo not in (0, 1):
            return RetCode.ERR_INVALIDPARAMETERS
        return RetCode.SUCCESS
