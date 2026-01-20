import numpy as np

from qnlab.problem.base import BaseProblem


class AckleyProblem(BaseProblem):
    """Ackley Problem

    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum_{i=1}^{n} x_i^2))
           - exp(1/n * sum_{i=1}^{n} cos(2 * pi * x_i))
           + 20 + e

    Global minimum:
        x^* = (0, ..., 0), f(x^*) = 0

    Reference:
        https://www.sfu.ca/~ssurjano/ackley.html

    Args:
        BaseProblem (_type_): _description_
    """

    def __init__(self, n: int = 100):
        np.random.seed(0)  # For reproducibility
        x0 = np.random.uniform(-32.768, 32.768, n).astype(np.float64)
        super().__init__("Ackley", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)

    def _f(self, x):
        # Ackley function value
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        term2 = -np.exp(sum_cos / n)
        return term1 + term2 + 20 + np.e

    def _g(self, x):
        # Gradient of the Ackley function
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))

        sqrt_sum_sq_n = np.sqrt(sum_sq / n)
        exp1 = np.exp(-0.2 * sqrt_sum_sq_n)
        exp2 = np.exp(sum_cos / n)

        # Avoid division by zero
        if sqrt_sum_sq_n == 0.0:
            grad1 = np.zeros_like(x)
        else:
            grad1 = (4.0 * x / (n * sqrt_sum_sq_n)) * exp1

        grad2 = (2.0 * np.pi / n) * np.sin(2 * np.pi * x) * exp2

        grad = grad1 + grad2
        return grad
