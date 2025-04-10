from qnlab.util.objectiveFunction import ObjectiveFunction
import numpy as np


class RosenbrockProblem(ObjectiveFunction):
    def __init__(self, n: int = 100):
        x0 = np.zeros(n)
        for i in range(n):
            x0[i] = 1.0 if i % 2 else -1.2
        super().__init__(n, x0)
        self.gnorms = []

    def evaluate(self, x):
        fx = 0.0
        grad = np.zeros_like(x)
        for i in range(0, self.n, 2):
            t1 = 1.0 - x[i]
            t2 = 10.0 * (x[i + 1] - x[i] ** 2)
            fx += t1**2 + t2**2
            grad[i + 1] = 20.0 * t2
            grad[i] = -2.0 * (x[i] * grad[i + 1] + t1)
        return fx, grad

    def progress(self, fx, gnorm, step, k):
        self._report_progress(fx, gnorm, step, k)
        self.gnorms.append(gnorm)
