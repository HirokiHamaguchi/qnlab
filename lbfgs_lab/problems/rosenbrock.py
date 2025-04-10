from lbfgs_lab.util.objectiveFunction import ObjectiveFunction
import numpy as np


class RosenbrockProblem(ObjectiveFunction):
    def __init__(self, n: int = 2):
        x0 = np.zeros(n)
        for i in range(n):
            x0[i] = 1.0 if i % 2 else -1.2
        super().__init__(n, x0)
        self.gnorms = []

    def evaluate(self, x):
        fx = 0.0
        grad = np.zeros_like(x)
        for i in range(len(x) - 1):
            temp = x[i + 1] - x[i] * x[i]
            fx += 100.0 * temp * temp + (1.0 - x[i]) ** 2
            grad[i] += -400.0 * x[i] * temp - 2.0 * (1.0 - x[i])
            grad[i + 1] += 200.0 * temp
        return fx, grad

    def progress(self, fx, gnorm, step, k):
        self._report_progress(fx, gnorm, step, k)
        self.gnorms.append(gnorm)
