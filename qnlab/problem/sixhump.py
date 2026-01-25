import numpy as np

from qnlab.problem.base import BaseProblem


class SixHumpProblem(BaseProblem):
    """Six-Hump Camelback Problem

    The objective function is:
    f(x, y) = (4 - 2.1*x^2 + x^4/3)*x^2 + xy + (-4 + 4*y^2)*y^2

    This function has 6 local minima and 2 global minima.
    Global minima are at approximately:
    (0.0898, -0.7126) and (-0.0898, 0.7126) with f(x*) ≈ -1.0316

    Reference: https://www.sfu.ca/~ssurjano/camel6.html
    """

    def __init__(self):
        x0 = np.array([-1.8, -0.8], dtype=np.float64)
        super().__init__("SixHump", 2, x0)
        # Set optimal point (one of the global minima)
        self.x_opt = np.array([0.0898, -0.7126], dtype=np.float64)

    def _f(self, x):
        """Compute the objective function value at x."""
        x0, x1 = x
        term1 = (4 - 2.1 * x0**2 + (x0**4) / 3.0) * x0**2
        term2 = x0 * x1
        term3 = (-4 + 4 * x1**2) * x1**2
        return term1 + term2 + term3

    def _g(self, x):
        """Compute the gradient at x."""
        x0, x1 = x
        grad_x0 = 8 * x0 - 8.4 * x0**3 + 2 * x0**5 + x1
        grad_x1 = x0 - 8 * x1 + 16 * x1**3
        return np.array([grad_x0, grad_x1], dtype=np.float64)

    def _hess(self, x):
        x0, x1 = x

        h00 = 8.0 - 25.2 * x0**2 + 10.0 * x0**4
        h01 = 1.0
        h11 = -8.0 + 48.0 * x1**2

        return np.array([[h00, h01], [h01, h11]], dtype=np.float64)

    def _hvp(self, x, v):
        H = self._hess(x)
        return H @ v
