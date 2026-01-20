import numpy as np

from qnlab.problem.base import BaseProblem


class BdqrticProblem(BaseProblem):
    """Bdqrtic Problem

    f(x) = \\sum_{i=1}^{n-4} (3 - 4 x_i)^2 + (x_i^2 + 2 x_{i+1}^2 + 3 x_{i+2}^2 + 4 x_{i+3}^2 + 5 x_n^2)^2

    Reference:
        https://www.cs.cas.cz/matonoha/download/V1081.pdf
    """

    def __init__(self, n: int = 5000):
        assert n >= 5, "n must be at least 5"
        x0 = np.ones(n)
        super().__init__("Bdqrtic", n, x0)

    def _f(self, x):
        x = np.asarray(x)
        n = self.n
        x0 = x[0 : n - 4]
        x1 = x[1 : n - 3]
        x2 = x[2 : n - 2]
        x3 = x[3 : n - 1]
        xn = x[-1]
        term1 = (3 - 4 * x0) ** 2
        term2 = (x0**2 + 2 * x1**2 + 3 * x2**2 + 4 * x3**2 + 5 * xn**2) ** 2
        return np.sum(term1 + term2)

    def _g(self, x):
        grad = np.zeros_like(x)

        x = np.asarray(x)
        n = self.n
        x0 = x[0 : n - 4]
        x1 = x[1 : n - 3]
        x2 = x[2 : n - 2]
        x3 = x[3 : n - 1]
        xn = x[-1]
        term = x0**2 + 2 * x1**2 + 3 * x2**2 + 4 * x3**2 + 5 * xn**2

        grad[0 : n - 4] -= 8 * (3 - 4 * x0)
        grad[0 : n - 4] += 4 * term * x0
        grad[1 : n - 3] += 8 * term * x1
        grad[2 : n - 2] += 12 * term * x2
        grad[3 : n - 1] += 16 * term * x3
        grad[-1] += np.sum(20 * term * xn)

        return grad
