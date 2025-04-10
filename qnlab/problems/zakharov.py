from qnlab.util.objectiveFunction import ObjectiveFunction
import numpy as np


class ZakharovProblem(ObjectiveFunction):
    def __init__(self, n: int = 100):
        x0 = np.ones(n, dtype=np.float64)
        super().__init__(n, x0)
        self.gnorms = []

    def evaluate(self, x):
        fx = 0.0
        grad = np.zeros_like(x)
        s = 0.0
        for i in range(len(x)):
            fx += x[i] * x[i]
            s += 0.5 * (i + 1) * x[i]
        fx += s * s + s**4
        for i in range(len(x)):
            ai = 0.5 * (i + 1)
            grad[i] = 2.0 * x[i] + (2.0 * s + 4.0 * (s**3)) * ai
        return fx, grad

    def progress(self, fx, gnorm, step, k):
        self.gnorms.append(gnorm)
