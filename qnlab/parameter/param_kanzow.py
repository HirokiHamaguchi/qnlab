from typing import Dict, Optional, Union

import numpy as np

from qnlab.parameter.param import BaseParameter


class KanzowParameter(BaseParameter):
    """Parameters for Kanzow wrapper."""

    def __init__(
        self, n: int, options: Optional[Dict[str, Union[np.float64, int]]] = None
    ) -> None:
        super().__init__(n, options=options)
