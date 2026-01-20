import numpy as np

from qnlab.problem.base import BaseProblem


class IllQuadraticProblem(BaseProblem):
    """Ill-Conditioned Quadratic Problem

    f(x) = 0.5 * x^T diag(1, ..., n) x

    x^* := argmin f(x) = (0, 0, ..., 0)  (f(x^*) = 0)
    """

    def __init__(self, n: int = 100):
        x0 = np.ones(n)
        super().__init__("IllQuadratic", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)
        self.diag = np.linspace(1, n, n, dtype=np.float64)

    def _f(self, x):
        return 0.5 * np.dot(x * self.diag, x)

    def _g(self, x):
        return self.diag * x

    def _hessian(self, x):
        return np.diag(self.diag)

    def _hvp(self, x, v):
        return self.diag * v
