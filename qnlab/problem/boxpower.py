import numpy as np

from qnlab.problem.base import BaseProblem


class BoxPowerProblem(BaseProblem):
    """BoxPower Problem

    f(x) = Σ_{i=1}^{N-1} (x_1 + x_i + x_N)^2 + (Σ x_i)^p

    Reference:
        https://bitbucket.org/optrove/sif/src/master/BOXPOWER.SIF
    """

    def __init__(self, n: int = 20000):
        assert n > 1, "n must be greater than 1"
        np.random.seed(0)  # For reproducibility
        x0 = np.ones(n) * 0.99
        self.p = 9
        super().__init__("Arwhead", n, x0)

    def _f(self, x) -> np.float64:
        term0 = x[0] ** 2
        term1 = np.sum((x[0] + x[1:-1] + x[-1]) ** 2)
        term2 = x[-1] ** (2 * self.p + 2)
        return term0 + term1 + term2

    def _g(self, x) -> np.ndarray:
        term0 = 2 * x[0]
        term1 = 2 * (x[0] + x[1:-1] + x[-1])
        term2 = (2 * self.p + 2) * x[-1] ** (2 * self.p + 1)
        sum_1 = np.sum(term1)
        grad = np.zeros_like(x)
        grad[0] = term0 + sum_1
        grad[1:-1] = term1
        grad[-1] = sum_1 + term2
        return grad
