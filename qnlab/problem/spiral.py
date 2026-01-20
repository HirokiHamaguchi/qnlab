import numpy as np

from qnlab.problem.base import BaseProblem


class SpiralProblem(BaseProblem):
    """Spiral Problem"""

    def __init__(self, n: int = 2):
        assert n == 2
        x0 = np.array([5, 5], dtype=np.float64)
        super().__init__("Spiral", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)

    def _f(self, x):
        x = x.copy()
        x[1] *= 5
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = np.arctan2(x[1], x[0])
        return 10 * np.sin(r - theta) + 0.01 * r**2 + 10

    def _g(self, x):
        # https://www.wolframalpha.com
        # gradient of 10 * sin(sqrt(x^2 + y^2) - arctan2(y, x)) + 0.01 * (x^2 + y^2)
        x = x.copy()
        x[1] *= 5
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        theta = np.arctan2(x[1], x[0])
        term1 = 10 * np.cos(r - theta) / (r**2) * (r * x + np.array([x[1], -x[0]]))
        term2 = 0.02 * x
        res = term1 + term2
        res[1] *= 5
        return res
