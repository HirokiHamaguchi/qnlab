import numpy as np

from qnlab.problem.base import BaseProblem


class PowellProblem(BaseProblem):
    """Powell Problem

    f(x) = sum_{i=1}^{n/4} [A_i^2 + 5*B_i^2 + C_i^4 + 10*D_i^4]
    A_i = x_{4i-3} + 10*x_{4i-2}
    B_i = x_{4i-1} - x_{4i}
    C_i = x_{4i-2} - 2*x_{4i-1}
    D_i = x_{4i-3} - x_{4i}

    x^* := argmin f(x) = (0,..., 0) (f(x^*) = 0)

    https://www.sfu.ca/~ssurjano/powell.html?utm_source=chatgpt.com

    Args:
        BaseProblem (_type_): _description_
    """

    def __init__(self, n: int = 100):
        if n % 4 != 0:
            raise ValueError("Powell function requires n to be a multiple of 4")
        x0 = np.array([3.0, -1.0, 0.0, 1.0] * (n // 4), dtype=np.float64)
        super().__init__("Powell", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)

    def _f(self, x):
        sum_val = np.float64(0.0)
        n_blocks = len(x) // 4
        for i in range(n_blocks):
            a = x[4 * i] + 10 * x[4 * i + 1]
            term1 = a**2
            b = x[4 * i + 2] - x[4 * i + 3]
            term2 = 5 * b**2
            c = x[4 * i + 1] - 2 * x[4 * i + 2]
            term3 = c**4
            d = x[4 * i] - x[4 * i + 3]
            term4 = 10 * d**4
            sum_val += term1 + term2 + term3 + term4
        return sum_val

    def _g(self, x):
        grad = np.zeros_like(x)
        n_blocks = len(x) // 4
        for i in range(n_blocks):
            idx = 4 * i
            a = x[idx] + 10 * x[idx + 1]
            b = x[idx + 2] - x[idx + 3]
            c = x[idx + 1] - 2 * x[idx + 2]
            d = x[idx] - x[idx + 3]
            # Derivatives from term1: a^2
            grad[idx] += 2 * a
            grad[idx + 1] += 20 * a
            # Derivatives from term2: 5*(b^2)
            grad[idx + 2] += 10 * b
            grad[idx + 3] += -10 * b
            # Derivatives from term3: c^4
            grad[idx + 1] += 4 * (c**3)
            grad[idx + 2] += -8 * (c**3)
            # Derivatives from term4: 10*(d^4)
            grad[idx] += 40 * (d**3)
            grad[idx + 3] += -40 * (d**3)
        return grad
