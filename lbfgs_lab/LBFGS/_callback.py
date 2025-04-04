import numpy as np

from lbfgs_lab.LBFGS._objectiveFunction import ObjectiveFunction


class CallbackData:
    def __init__(
        self,
        n: int,
        instance: ObjectiveFunction,
    ):
        self.n = n
        self.instance = instance


class IterationData:
    def __init__(self, n: int):
        self.alpha = 0.0
        self.s = np.zeros(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.ys = 0.0  # inner product of y and s
