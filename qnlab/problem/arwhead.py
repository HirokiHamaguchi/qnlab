import numpy as np

from qnlab.problem.base import BaseProblem


class ArwheadProblem(BaseProblem):
    """Arwhead Problem

    f(x) = \\sum_{i=1}^{n-1} ((x_i^2+x_n^2)^2 - 4 x_i + 3)

    Reference:
        https://www.cs.cas.cz/matonoha/download/V1081.pdf
    """

    def __init__(self, n: int = 5000):
        x0 = np.ones(n)
        super().__init__("Arwhead", n, x0)

    def _f(self, x):
        # Ackley function value
        return np.sum((x[:-1] ** 2 + x[-1] ** 2) ** 2 - 4 * x[:-1] + 3)

    def _g(self, x):
        # Gradient of the Ackley function
        grad = np.zeros_like(x)
        grad[:-1] = 4 * (x[:-1] ** 2 + x[-1] ** 2) * x[:-1] - 4
        grad[-1] = 4 * np.sum(x[:-1] ** 2 + x[-1] ** 2) * x[-1]
        return grad
