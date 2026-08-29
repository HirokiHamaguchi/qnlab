import numpy as np

from qnlab.util.ret_values import RetCode


class BaseParameter:
    """Shared parameters across solvers."""

    def __init__(
        self, n: int, options: dict[str, np.float64 | int] | None = None
    ) -> None:
        # Universal parameters
        self.m: int = 10
        self.gtol: np.float64 = np.float64(1e-5)
        self.max_iterations: int = 15000
        self.max_evaluations: int = 30000
        self.eps: np.float64 = np.float64(2.2204460492503131e-09)

        if options:
            self._apply_options(options)

        self._validate_shared()

    def __str__(self) -> str:  # pragma: no cover - readability helper
        return f"BaseParameter(m={self.m}, gtol={self.gtol})"

    def _apply_options(self, options: dict[str, np.float64 | int]) -> None:
        for key, value in options.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                import difflib

                closest_match = difflib.get_close_matches(
                    key, self.__dict__.keys(), n=3, cutoff=0.6
                )
                closest_mes = (
                    f" Did you mean one of: {', '.join(closest_match)}?"
                    if closest_match
                    else ""
                )
                raise ValueError(f"Unknown parameter '{key}'.{closest_mes}")

    def _validate_shared(self) -> None:
        code = self._check_shared()
        if code.is_error():
            raise ValueError(str(code))

    def _check_shared(self) -> RetCode:
        if (self.m <= 0 and self.m != -1) or 1e10 <= self.m:
            return RetCode.ERR_INVALID_M
        if self.gtol < 0.0:
            return RetCode.ERR_INVALID_GTOL
        return RetCode.SUCCESS
