import numpy as np

from qnlab.parameter.param_line import LineParameter
from qnlab.util.ret_values import RetCode


class OwlParameter(LineParameter):
    """Parameters for OWL-QN solvers."""

    def __init__(
        self, n: int, options: dict[str, np.float64 | int] | None = None
    ) -> None:
        self.orthantwise_c: np.float64 = np.float64(0.0)
        self.orthantwise_start: int = 0
        self.orthantwise_end: int = n
        self.n = n
        super().__init__(n, options=None)

        if options:
            self._apply_options(options)
        self._validate()

        from qnlab.util.linesearch import line_search_backtracking_owlqn

        self.linesearch = line_search_backtracking_owlqn

    def _validate(self) -> None:
        code = self._check_shared()
        if code.is_error():
            raise ValueError(str(code))
        code = self._check_line_params()
        if code.is_error():
            raise ValueError(str(code))
        code = self._check_owl_params()
        if code.is_error():
            raise ValueError(str(code))

    def _check_owl_params(self) -> RetCode:
        from qnlab.util.linesearch import LINESEARCH_BACKTRACKING

        if self.orthantwise_c < 0.0:
            return RetCode.ERR_INVALID_ORTHANTWISE
        if self.orthantwise_start < 0 or self.n < self.orthantwise_start:
            return RetCode.ERR_INVALID_ORTHANTWISE_START
        if self.orthantwise_end < 0 or self.n < self.orthantwise_end:
            return RetCode.ERR_INVALID_ORTHANTWISE_END
        if (
            self.orthantwise_c != 0.0
            and self.linesearch_kind != LINESEARCH_BACKTRACKING
        ):
            return RetCode.ERR_INVALID_LINESEARCH
        return RetCode.SUCCESS
