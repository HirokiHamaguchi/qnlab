import numpy as np

from qnlab.util.objectiveFunction import ObjectiveFunction


class CallbackData:
    """Stores data passed to the callback during optimization."""

    def __init__(self, n: int, instance: ObjectiveFunction):
        """Initializes the CallbackData.

        Args:
            n (int): Number of variables.
            instance (ObjectiveFunction): Objective function instance.
        """
        self.n = n
        self.instance = instance


class IterationData:
    """Data structure to store per-iteration vectors and scalars."""

    def __init__(self, n: int):
        """Initializes the IterationData.

        Args:
            n (int): Dimension for s and y vectors.
        """
        self.alpha = 0.0
        self.s = np.zeros(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.ys = 0.0  # inner product of y and s
