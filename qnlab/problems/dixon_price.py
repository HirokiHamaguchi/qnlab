from qnlab.util.objectiveFunction import ObjectiveFunction
import numpy as np


class DixonPriceProblem(ObjectiveFunction):
    def __init__(self, n: int = 100):
        x0 = np.full(n, 0.5, dtype=np.float64)
        super().__init__(n, x0)
        self.gnorms = []

    def evaluate(self, x):
        fx = 0.0
        grad = np.zeros_like(x)
        fx += (x[0] - 1.0) ** 2
        grad[0] = 2.0 * (x[0] - 1.0)
        for i in range(1, len(x)):
            temp = 2.0 * x[i] * x[i] - x[i - 1]
            fx += i * temp * temp
            grad[i] += 4.0 * i * x[i] * temp
            grad[i - 1] -= 2.0 * i * temp
        return fx, grad

    def progress(self, fx, gnorm, step, k):
        self.gnorms.append(gnorm)
