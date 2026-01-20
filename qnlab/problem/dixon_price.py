import numpy as np

from qnlab.problem.base import BaseProblem


class DixonPriceProblem(BaseProblem):
    """Dixon-Price Problem

    f(x) = (x_1 - 1)^2 + sum_{i=2}^{n} i * (2 * x_i^2 - x_{i-1})^2

    x^* := argmin f(x) = (2^{-(2^i-2)/(2^i)}) (1-indexed)  (f(x^*) = 0)

    https://www.sfu.ca/~ssurjano/dixonpr.html

    Note:
        This is a non-convex problem.
    """

    def __init__(self, n: int = 100):
        np.random.seed(0)
        x0 = np.random.uniform(-5.0, 5.0, n).astype(np.float64)
        super().__init__("DixonPrice", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)
        for i in range(n):
            self.x_opt[i] = 2.0 ** (-(2 ** (i + 1) - 2) / (2 ** (i + 1)))
        assert np.isclose(self._f(self.x_opt), 0.0)

    def _f(self, x):
        fx = (x[0] - 1.0) ** 2
        for i in range(1, len(x)):
            temp = 2.0 * x[i] * x[i] - x[i - 1]
            fx += i * temp * temp
        return fx

    def _g(self, x):
        grad = np.zeros_like(x)
        grad[0] = 2.0 * (x[0] - 1.0)
        for i in range(1, len(x)):
            temp = 2.0 * x[i] * x[i] - x[i - 1]
            grad[i] += 8.0 * i * x[i] * temp
            grad[i - 1] -= 2.0 * i * temp
        return grad
