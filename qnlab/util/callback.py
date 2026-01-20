import time
from typing import DefaultDict, List

import numpy as np
import numpy.typing as npt

from qnlab.problem.base import BaseProblem


class Callback:
    def __init__(
        self,
        gnorm_order=np.inf,
        save_xs=False,
    ) -> None:
        """Initializes the Callback.

        Args:
            gnorm_order: norm ord of gnorm
            save_xs (bool): If True, saves the x vectors.
        """
        # SciPy also uses infinity norm for gnorm
        self.gnorm_order = gnorm_order
        self.save_xs = save_xs
        self.reset()

    def reset(self) -> None:
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
    ) -> None:
        """Called at the start of the optimization."""
        prob.reset()
        self.reset()
        self.callback(
            prob,
            prob.x0,
            prob.f(x0, count=False),
            prob.g(x0, count=False),
        )

    def callback(
        self,
        prob: BaseProblem,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        g: npt.NDArray[np.float64],
    ) -> None:
        if self.save_xs:
            self.xs.append(np.copy(x))

        gnorm = np.float64(np.linalg.norm(g, ord=self.gnorm_order))
        self.fxs.append(fx)
        self.gnorms.append(gnorm)
        self.times.append(time.perf_counter() - self.start_time)
        self.calls.append(prob.count_calls())
