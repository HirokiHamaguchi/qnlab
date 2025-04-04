import numpy as np

from lbfgs_lab.LBFGS._objectiveFunction import ObjectiveFunction


class CallbackData:
    """
    Stores data passed to the callback during optimization.
    """

    def __init__(
        self,
        n: int,
        instance: ObjectiveFunction,
    ):
        """
        :param n: Number of variables
        :param instance: Objective function instance
        """
        self.n = n
        self.instance = instance


class IterationData:
    """
    Data structure to store per-iteration vectors and scalars.
    """

    def __init__(self, n: int):
        """
        :param n: Dimension for s and y vectors
        """
        self.alpha = 0.0
        self.s = np.zeros(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.ys = 0.0  # inner product of y and s
