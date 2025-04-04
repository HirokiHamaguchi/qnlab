import numpy as np
import numpy.typing as npt
from typing import Tuple


class ObjectiveFunction:
    """Example objective function for optimization tests."""

    def __init__(self):
        """Initializes the objective function."""
        pass

    def evaluate(
        self, x: npt.NDArray[np.float64]
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluates the function value and gradient at x.

        Args:
            x (numpy.ndarray): Input variable vector.

        Returns:
            Tuple[float, numpy.ndarray]: Function value and gradient.
        """
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
        """Prints iteration progress.

        Args:
            fx (float): Current function value.
            gnorm (float): Norm of the gradient.
            step (float): Step size.
            k (int): Iteration index.
        """
        print(f"Iteration {k}:")
        print(f" fx={fx:.6f} gnorm={gnorm:.6f} step={step:.6f}")
        print()
