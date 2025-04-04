import numpy as np
import numpy.typing as npt
from typing import Tuple


class ObjectiveFunction:
    def __init__(self):
        pass

    def evaluate(
        self, x: npt.NDArray[np.float64]
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        fx = 0.0
        g = np.zeros_like(x)
        for i in range(0, len(x), 2):
            t1 = 1.0 - x[i]
            t2 = 10.0 * (x[i + 1] - x[i] * x[i])
            g[i + 1] = 20.0 * t2
            g[i] = -2.0 * (x[i] * g[i + 1] + t1)
            fx += t1 * t1 + t2 * t2
        return fx, g

    def progress(
        self,
        fx: float,
        gnorm: float,
        step: float,
        k: int,
    ) -> None:
        print(f"Iteration {k}:")
        print(f" fx={fx:.6f} gnorm={gnorm:.6f} step={step:.6f}")
        print()
