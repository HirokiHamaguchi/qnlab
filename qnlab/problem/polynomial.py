import numpy as np

from qnlab.problem.base import BaseProblem


class PolynomialProblem(BaseProblem):
    """Polynomial Problem"""

    def __init__(self, n: int = 1, c_4=1.0, c_3=0.0, c_2=0.0, c_1=0.0, c_0=0.0):
        assert n == 1
        x0 = np.array([3.0] * n, dtype=np.float64)
        self.c0 = c_0
        self.c1 = c_1
        self.c2 = c_2
        self.c3 = c_3
        self.c4 = c_4
        super().__init__("Polynomial", n, x0)

    def _f(self, x: np.ndarray):
        x0 = x[0]
        return (
            self.c4 * x0**4 + self.c3 * x0**3 + self.c2 * x0**2 + self.c1 * x0 + self.c0
        )

    def _g(self, x: np.ndarray):
        """Gradient of the function."""
        x0 = x[0]
        return np.array(
            [
                4 * self.c4 * x0**3 + 3 * self.c3 * x0**2 + 2 * self.c2 * x0 + self.c1,
            ]
        )
