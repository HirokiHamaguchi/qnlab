import numpy as np
import numpy.typing as npt
from typing import Tuple, Union


class ObjectiveFunction:
    """Example objective function for optimization tests."""

    n: int  # Number of variables
    x0: npt.NDArray[np.float64]  # Initial point

    def __init__(self, n: int = 0, x0: Union[npt.NDArray[np.float64], None] = None):
        """Initializes the objective function."""
        self.n = n
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = np.zeros(n, dtype=np.float64)

    def evaluate(
        self, x: npt.NDArray[np.float64]
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluates the function value and gradient at x.

        Args:
            x (numpy.ndarray): Input variable vector.

        Returns:
            Tuple[float, numpy.ndarray]: Function value and gradient.
        """
        return 0.0, np.zeros_like(x)

    def _report_progress(
        self,
        fx: float,
        gnorm: float,
        step: float,
        k: int,
    ) -> None:
        """Reports the progress of the optimization."""
        print(f"Iteration {k}:")
        print(f" fx={fx:.6f} gnorm={gnorm:.6f} step={step:.6f}")
        print()

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
        self._report_progress(fx, gnorm, step, k)
