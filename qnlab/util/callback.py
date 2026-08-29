import time
from typing import DefaultDict, List, Optional

import numpy as np
import numpy.typing as npt
from qnlab.problem.base import BaseProblem


class CallbackTimeoutError(RuntimeError):
    """Raised when the callback exceeds a prescribed time limit."""


class Callback:
    def __init__(
        self,
        gnorm_order=np.inf,
        save_xs=False,
        time_limit: Optional[float] = None,
    ) -> None:
        """Initializes the Callback.

        Args:
            gnorm_order: norm ord of gnorm
            save_xs (bool): If True, saves the x vectors.
        """
        # SciPy also uses infinity norm for gnorm
        self.gnorm_order = gnorm_order
        self.save_xs = save_xs
        self.time_limit = time_limit
        self._reset()

    def __repr__(self) -> str:
        if len(self.fxs) >= 2:
            return f"Callback(iteration={self.iteration}, fxs=({self.fxs[0]},...,{self.fxs[-1]}) (total {len(self.fxs)}))"
        else:
            return f"Callback(iteration={self.iteration}, fxs={self.fxs})"

    def _reset(self) -> None:
        """Resets the callback data."""
        self.xs: List[npt.NDArray[np.float64]] = []
        self.fxs: List[np.float64] = []
        self.gnorms: List[np.float64] = []
        self.times: List[float] = []
        self.calls: List[int] = []
        self.others: DefaultDict[str, int] = DefaultDict(int)
        self.start_time: float = time.perf_counter()
        self.iteration: int = 0

    def start(
        self,
        prob: BaseProblem,
        x0: npt.NDArray[np.float64],
        gnorm_vector: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Called at the start of the optimization."""
        prob.reset()
        self._reset()
        self.callback(
            prob,
            x0,
            prob.f(x0, count=False),
            prob.g(x0, count=False),
            gnorm_vector=gnorm_vector,
        )

    def callback(
        self,
        prob: BaseProblem,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        g: npt.NDArray[np.float64],
        gnorm_vector: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        if self.save_xs:
            self.xs.append(np.copy(x))

        vector = g if gnorm_vector is None else gnorm_vector
        gnorm = np.float64(np.linalg.norm(vector, ord=self.gnorm_order))
        self.fxs.append(fx)
        self.gnorms.append(gnorm)

        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)
        self.calls.append(prob.count_calls())

        if self.time_limit is not None and elapsed >= self.time_limit:
            raise CallbackTimeoutError(
                f"Elapsed time {elapsed:.2f}s exceeded limit {self.time_limit:.2f}s"
            )
