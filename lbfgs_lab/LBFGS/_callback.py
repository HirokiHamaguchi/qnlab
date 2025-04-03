import numpy as np
import numpy.typing as npt
from typing import Callable, TypeAlias

from lbfgs_lab.LBFGS._objectiveFunction import ObjectiveFunction

lbfgs_evaluate_t: TypeAlias = Callable[
    [
        npt.NDArray[np.float64],  # x
        npt.NDArray[np.float64],  # g
        int,  # n
        float,  # step
    ],
    float,  # objective function value
]

lbfgs_progress_t = Callable[
    [
        ObjectiveFunction,  # instance
        npt.NDArray[np.float64],  # x
        npt.NDArray[np.float64],  # g
        float,  # fx
        float,  # xnorm
        float,  # gnorm
        float,  # step
        int,  # n
        int,  # k
        int,  # ls
    ],
    int,  # zero to continue, non-zero to stop
]


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
