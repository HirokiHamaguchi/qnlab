from collections import deque
from typing import Iterator, Optional

import numpy as np
import numpy.typing as npt

from qnlab.util.callback import Callback
from qnlab.util.iteration_data import IterationData
from qnlab.util.method import Method


class QuasiNewtonMemory:
    """Limited-memory container used by quasi-Newton methods."""

    def __init__(self, g: np.ndarray, maxlen: int, method: Method) -> None:
        self._deque: deque[IterationData] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._method = method
        gnorm = np.linalg.norm(g)
        self.zero_length = 1 / gnorm if gnorm > 0 else 1.0

    def __iter__(self) -> Iterator[IterationData]:
        return iter(self._deque)

    def __reversed__(self) -> Iterator[IterationData]:
        return reversed(self._deque)

    def __len__(self) -> int:
        return len(self._deque)

    def add_new_data(
        self,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        g: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fxp: np.float64,
        gp: npt.NDArray[np.float64],
        callback: Optional[Callback],
        eps: np.float64,
    ) -> None:
        new_data = IterationData()
        is_valid, message = new_data.set(x, fx, g, xp, fxp, gp, self._method, eps)
        if not is_valid:
            if callback is not None:
                callback.others[message] += 1
            return
        assert all(np.isfinite(val) for val in (new_data.ss, new_data.ys, new_data.yy))
        self._deque.append(new_data)

    def get_last(self) -> IterationData:
        if not self._deque:
            raise RuntimeError("Memory is empty.")
        return self._deque[-1]

    def zero_memory_direction(self, g: np.ndarray, mu: np.float64) -> np.ndarray:
        if mu > 0.0:
            return -g / mu
        else:
            return -g * self.zero_length
