from typing import Dict, Optional, Union

import numpy as np

from qnlab.parameter.param import BaseParameter


class NtqnParameter(BaseParameter):
    """Parameters for NTQN wrapper."""

    def __init__(
        self, n: int, options: Optional[Dict[str, Union[np.float64, int]]] = None
    ) -> None:
        super().__init__(n, options=None)
        self.terminate: int = 1
        self.stop_at_gtol: int = 1

        if options:
            self._apply_options(options)
            self._validate_shared()

        if self.terminate not in (0, 1, 2, 3):
            raise ValueError("terminate must be one of {0, 1, 2, 3}.")
        if self.stop_at_gtol not in (0, 1):
            raise ValueError("stop_at_gtol must be either 0 or 1.")
