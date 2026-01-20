import numpy as np

from qnlab.problem.base import BaseProblem


class NonConvexProblem(BaseProblem):
    """Non-Convex Problem"""

    def __init__(self, n: int = 1):
        assert n == 1, "Non Convex is defined for n=1 only."
        x0 = np.array([np.sqrt(2)], dtype=np.float64)
        super().__init__("NonConvex", n, x0)

    def _f(self, x):
        return np.float64(1e5) - np.linalg.norm(x) ** 2 + 1e-5 * np.linalg.norm(x) ** 3

    def _g(self, x):
        return -2 * x + 3e-5 * np.linalg.norm(x) * x
