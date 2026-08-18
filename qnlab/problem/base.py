from abc import ABC, abstractmethod
from typing import Union, final

import numpy as np
import numpy.typing as npt


class BaseProblem(ABC):
    """Base class for optimization problems."""

    name: str
    n: int  # Number of variables
    x0: npt.NDArray[np.float64]  # Initial point
    x_opt: npt.NDArray[np.float64]  # Optimal point, maybe None
    call_f: int  # Number of function evaluations
    call_g: int  # Number of gradient evaluations
    call_hvp: int  # Number of Hessian-vector product evaluations

    def __init__(
        self,
        name: str,
        n: int = 0,
        x0: Union[npt.NDArray[np.float64], None] = None,
    ):
        """Initializes the objective function."""
        self.name = name
        self.n = n
        self.x0 = x0 if x0 is not None else np.zeros(n, dtype=np.float64)
        assert self.x0.shape == (self.n,), "x0 must have shape (n,)"
        self.reset()

    def __repr__(self):
        return (
            self.name
            + f" (n={self.n}, {self.call_f=}, {self.call_g=}, {self.call_hvp=})"
        )

    @final
    def reset(self):
        """Resets the function evaluation counters."""
        self.call_f = 0
        self.call_g = 0
        self.call_hvp = 0

    @final
    def f(self, x: npt.NDArray[np.float64], count: bool = True) -> np.float64:
        """Compute the objective function value at x."""
        if count:
            self.call_f += 1
        return self._f(x)

    @final
    def g(
        self, x: npt.NDArray[np.float64], count: bool = True
    ) -> npt.NDArray[np.float64]:
        """Compute the gradient at x."""
        if count:
            self.call_g += 1
        return self._g(x)

    @final
    def hvp(
        self, x: npt.NDArray[np.float64], v: npt.NDArray[np.float64], count: bool = True
    ) -> npt.NDArray[np.float64]:
        """Compute the Hessian-vector product at x with vector v."""
        if count:
            self.call_hvp += 1
        return self._hvp(x, v)

    @final
    def count_calls(self) -> int:
        """Returns the total number of function evaluations."""
        return self.call_f + self.call_g + self.call_hvp

    # ------ Abstract methods ------ (Will be implemented by subclasses)

    @abstractmethod
    def _f(self, x: npt.NDArray[np.float64]) -> np.float64:
        """Compute the objective function value at x."""
        pass

    @abstractmethod
    def _g(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the gradient at x."""
        pass

    def _hessian(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the Hessian matrix at x."""
        raise NotImplementedError("Hessian not implemented for this problem.")

    def _hvp(
        self, x: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the Hessian-vector product at x with vector v."""
        return self._hessian(x) @ v

    def get_machine_eps(self) -> np.float64:
        """Returns the machine epsilon for this problem.

        Default value is for 64-bit precision: 2.2204460492503131e-09
        """
        return np.float64(np.finfo(np.float64).eps * 1e7)

    def get_noise(self) -> np.float64 | npt.NDArray[np.float64]:
        """Returns the noise level to gradient for this problem.

        Default value is 0.0 (no noise).
        """
        return np.float64(0.0)
