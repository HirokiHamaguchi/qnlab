import numpy as np

from qnlab.problem.base import BaseProblem


class StairProblem(BaseProblem):
    """Stair Problem"""

    def __init__(self, n: int = 1, a: float = 1):
        assert n == 1, "Stair Problem is defined for n=1 only."
        x0 = np.array([95 * a], dtype=np.float64)
        self.x_ks = np.arange(10, 100, 10) * a
        super().__init__("Stair", n, x0)
        self.x_opt = np.zeros(n, dtype=np.float64)

    def _f(self, x):
        term1 = np.sum([1 / (1 + np.exp(-x + x_k)) for x_k in self.x_ks])
        term2 = np.sum([1 / (1 + np.exp(+x + x_k)) for x_k in self.x_ks])
        # xが100の時点で、1になる程度の補正
        return term1 + term2 + 1e-4 * (x[0] ** 2)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _g(self, x):
        term1 = np.sum(
            [
                self.sigmoid(+x - x_k) * (1 - self.sigmoid(+x - x_k))
                for x_k in self.x_ks
            ],
            axis=0,
        )
        term2 = np.sum(
            [
                self.sigmoid(-x - x_k) * (1 - self.sigmoid(-x - x_k))
                for x_k in self.x_ks
            ],
            axis=0,
        )
        return term1 - term2 + 2e-4 * x
