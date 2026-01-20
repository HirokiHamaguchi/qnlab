import numpy as np
import scipy.optimize

from qnlab.problem.base import BaseProblem


class RosenbrockProblem(BaseProblem):
    """Rosenbrock Problem

    f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    x^* := argmin f(x) = (1, 1, ..., 1)  (f(x^*) = 0)

    https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(self, n: int = 100):
        x0 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x0[i] = 1.0 if i % 2 else -1.2
        super().__init__("Rosenbrock", n, x0)
        self.x_opt = np.ones(n, dtype=np.float64)

    # To avoid machine precision issues,
    # We don't use scipy.optimize.rosen
    def _f(self, x):
        temp = x[1:] - x[:-1] * x[:-1]
        r = np.sum(100.0 * temp * temp + (1.0 - x[:-1]) * (1.0 - x[:-1]))
        return r

    def _g(self, x):
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x, dtype=np.float64)
        der[1:-1] = (
            200.0 * (xm - xm_m1 * xm_m1)
            - 400.0 * (xm_p1 - xm * xm) * xm
            - 2.0 * (1.0 - xm)
        )
        der[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])
        der[-1] = 200.0 * (x[-1] - x[-2] * x[-2])
        return der

    def _hessian(self, x):
        return np.array(scipy.optimize.rosen_hess(x))

    def _hvp(self, x, v):
        """Hessian-vector product"""
        return scipy.optimize.rosen_hess_prod(x, v)
