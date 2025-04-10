from qnlab.util.objectiveFunction import ObjectiveFunction
import numpy as np


class PowellProblem(ObjectiveFunction):
    def __init__(self):
        x0 = np.array([3.0, -1.0, 0.0, 1.0], dtype=np.float64)
        super().__init__(4, x0)
        self.gnorms = []

    def evaluate(self, x):
        if len(x) != 4:
            raise ValueError("Powell function requires dimension 4")
        A = x[0] + 10.0 * x[1]
        B = x[2] - x[3]
        C = x[1] - 2.0 * x[2]
        D = x[0] - x[3]
        fx = A * A + 5.0 * B * B + C**4 + 10.0 * D**4
        grad = np.zeros_like(x)
        grad[0] += 2.0 * A
        grad[1] += 20.0 * A
        grad[2] += 10.0 * B
        grad[3] -= 10.0 * B
        C3 = 4.0 * (C**3)
        grad[1] += C3
        grad[2] -= 2.0 * C3
        D3 = 40.0 * (D**3)
        grad[0] += D3
        grad[3] -= D3
        return fx, grad

    def progress(self, fx, gnorm, step, k):
        self.gnorms.append(gnorm)
