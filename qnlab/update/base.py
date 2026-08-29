from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from qnlab.util.memory_interface import QuasiNewtonMemory


class BaseUpdateRule(ABC):
    @staticmethod
    @abstractmethod
    def compute_dir(
        x: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64],  # grad f(x)
        lm: QuasiNewtonMemory,
    ) -> npt.NDArray[np.float64]:
        """Compute the search direction."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    @staticmethod
    @abstractmethod
    def compute_dir_reg(
        x: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64],  # grad f(x)
        lm: QuasiNewtonMemory,
        mu: np.float64,  # Regularized parameter
    ) -> npt.NDArray[np.float64]:
        """Compute the search direction with regularization."""
        raise NotImplementedError("The regularized version is not implemented.")

    @staticmethod
    @abstractmethod
    def check(
        n: int,
        g: npt.NDArray[np.float64],  # Gradient vector
        d: npt.NDArray[np.float64],  # Search direction
        lm: QuasiNewtonMemory,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Perform a check on the update."""
