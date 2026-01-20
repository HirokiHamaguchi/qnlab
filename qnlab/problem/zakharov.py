import numpy as np

from qnlab.problem.base import BaseProblem


class ZakharovProblem(BaseProblem):
    """Zakharov Problem

    f(x) = sum_{i=1}^{n} x_i^2
        + (sum_{i=1}^{n} 0.5 i x_i)^2
        + (sum_{i=1}^{n} 0.5 i x_i)^4

    x^* := argmin f(x) = (0, 0, ..., 0)  (f(x^*) = 0)

    https://www.sfu.ca/~ssurjano/zakharov.html
    """

    def __init__(self, n: int = 100):
        x0 = np.ones(n, dtype=np.float64)
        super().__init__("Zakharov", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)

    def _f(self, x):
        fx = np.float64(0.0)
        s = 0.0
        for i in range(len(x)):
            fx += x[i] * x[i]
            s += 0.5 * (i + 1) * x[i]
        return fx + s * s + s**4

    def _g(self, x):
        grad = np.zeros_like(x)
        s = 0.0
        for i in range(len(x)):
            s += 0.5 * (i + 1) * x[i]
        for i in range(len(x)):
            ai = 0.5 * (i + 1)
            grad[i] = 2.0 * x[i] + (2.0 * s + 4.0 * (s**3)) * ai
        return grad
