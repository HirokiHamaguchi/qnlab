from collections import deque
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

from qnlab.util.callback import Callback
from qnlab.util.iteration_data import IterationData
from qnlab.util.method import Method


class LBFGSWorkspace:
    """Contiguous L-BFGS vectors and incrementally maintained inner products."""

    def __init__(self, n: int, capacity: int) -> None:
        self.capacity = capacity
        self.size = 0
        self._start = 0
        # Columns are traversed repeatedly by the L-BFGS two-loop recursion.
        self._steps = np.empty((n, capacity), dtype=np.float64, order="F")
        self._gradients = np.empty((n, capacity), dtype=np.float64, order="F")
        self._step_products = np.empty((capacity, capacity), dtype=np.float64)
        self._step_gradient = np.empty((capacity, capacity), dtype=np.float64)
        self._gradient_products = np.empty((capacity, capacity), dtype=np.float64)
        self._step_norms = np.empty(capacity, dtype=np.float64)
        self._pair_products = np.empty(capacity, dtype=np.float64)
        self._gradient_norms = np.empty(capacity, dtype=np.float64)
        self._grams_initialized = False
        self.alphas = np.empty(capacity, dtype=np.float64)

    @property
    def steps(self) -> npt.NDArray[np.float64]:
        if self._start == 0:
            return self._steps[:, : self.size]
        return self._steps[:, self.indices]

    @property
    def gradients(self) -> npt.NDArray[np.float64]:
        if self._start == 0:
            return self._gradients[:, : self.size]
        return self._gradients[:, self.indices]

    @property
    def step_products(self) -> npt.NDArray[np.float64]:
        self._ensure_grams()
        return self._ordered_square(self._step_products)

    @property
    def step_gradient(self) -> npt.NDArray[np.float64]:
        self._ensure_grams()
        return self._ordered_square(self._step_gradient)

    @property
    def gradient_products(self) -> npt.NDArray[np.float64]:
        self._ensure_grams()
        return self._ordered_square(self._gradient_products)

    @property
    def step_norms(self) -> npt.NDArray[np.float64]:
        return self._step_norms[self.indices]

    @property
    def pair_products(self) -> npt.NDArray[np.float64]:
        return self._pair_products[self.indices]

    @property
    def gradient_norms(self) -> npt.NDArray[np.float64]:
        return self._gradient_norms[self.indices]

    @property
    def indices(self) -> npt.NDArray[np.intp]:
        """Physical column indices in oldest-to-newest order."""
        if self.capacity == 0:
            return np.empty(0, dtype=np.intp)
        return (self._start + np.arange(self.size, dtype=np.intp)) % self.capacity

    @property
    def last_index(self) -> int:
        if self.size == 0:
            raise RuntimeError("Workspace is empty.")
        return int((self._start + self.size - 1) % self.capacity)

    def _ordered_square(
        self, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        if self._start == 0:
            return matrix[: self.size, : self.size]
        indices = self.indices
        return matrix[np.ix_(indices, indices)]

    def _ensure_grams(self) -> None:
        if self._grams_initialized:
            return
        # Gram matrices use physical ring-buffer indices internally.
        steps = self._steps[:, : self.size]
        gradients = self._gradients[:, : self.size]
        self._step_products[: self.size, : self.size] = steps.T @ steps
        self._step_gradient[: self.size, : self.size] = steps.T @ gradients
        self._gradient_products[: self.size, : self.size] = gradients.T @ gradients
        self._grams_initialized = True

    def append(
        self,
        step: npt.NDArray[np.float64],
        gradient_difference: npt.NDArray[np.float64],
        ss: np.float64 | None = None,
        ys: np.float64 | None = None,
        yy: np.float64 | None = None,
    ) -> None:
        """Append one pair while updating all Gram matrices in ``O(n*m)``."""
        if self.capacity == 0:
            return
        if self.size == self.capacity:
            # Overwrite the oldest column instead of shifting all n*m entries.
            index = self._start
            self._start = (self._start + 1) % self.capacity
        else:
            index = (self._start + self.size) % self.capacity
            self.size += 1

        self._steps[:, index] = step
        self._gradients[:, index] = gradient_difference
        self._step_norms[index] = np.dot(step, step) if ss is None else ss
        self._pair_products[index] = (
            np.dot(step, gradient_difference) if ys is None else ys
        )
        self._gradient_norms[index] = (
            np.dot(gradient_difference, gradient_difference) if yy is None else yy
        )
        if not self._grams_initialized:
            return

        # Before the buffer is full the active physical columns are [0, size).
        # Once full, every physical column is active irrespective of logical order.
        steps = self._steps[:, : self.size]
        gradients = self._gradients[:, : self.size]

        step_column = steps.T @ step
        self._step_products[: self.size, index] = step_column
        self._step_products[index, : self.size] = step_column

        gradient_column = gradients.T @ gradient_difference
        self._gradient_products[: self.size, index] = gradient_column
        self._gradient_products[index, : self.size] = gradient_column

        self._step_gradient[: self.size, index] = steps.T @ gradient_difference
        self._step_gradient[index, : self.size] = step @ gradients

    def rebuild(self, items: Iterator[IterationData]) -> None:
        """Rebuild after an explicit memory removal."""
        self.size = 0
        self._start = 0
        self._grams_initialized = False
        for item in items:
            self.append(item.s, item.y, item.ss, item.ys, item.yy)


class QuasiNewtonMemory:
    """Limited-memory container used by quasi-Newton methods."""

    def __init__(
        self,
        g: np.ndarray,
        maxlen: int,
        method: Method,
        zero_regularized_hessian_scale: np.float64 | None = None,
        curvature_scale_floor: np.float64 | None = None,
    ) -> None:
        self._deque: deque[IterationData] = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._method = method
        self.curvature_scale_floor = curvature_scale_floor
        self.workspace = LBFGSWorkspace(g.size, maxlen)
        gnorm = np.linalg.norm(g)
        self.zero_hessian_scale = np.float64(gnorm if gnorm > 0 else 1.0)
        self.zero_regularized_hessian_scale = (
            self.zero_hessian_scale
            if zero_regularized_hessian_scale is None
            else np.float64(zero_regularized_hessian_scale)
        )
        if self.zero_regularized_hessian_scale <= 0.0:
            raise ValueError("zero_regularized_hessian_scale must be positive")

    def __iter__(self) -> Iterator[IterationData]:
        return iter(self._deque)

    def __reversed__(self) -> Iterator[IterationData]:
        return reversed(self._deque)

    def __len__(self) -> int:
        return len(self._deque)

    def add_new_data(
        self,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        g: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fxp: np.float64,
        gp: npt.NDArray[np.float64],
        callback: Callback | None,
        eps: np.float64,
    ) -> None:
        new_data = IterationData()
        is_valid, message = new_data.set(
            x,
            fx,
            g,
            xp,
            fxp,
            gp,
            self._method,
            eps,
            self.curvature_scale_floor,
        )
        if not is_valid:
            if callback is not None:
                callback.others[message] += 1
            return
        assert all(np.isfinite(val) for val in (new_data.ss, new_data.ys, new_data.yy))
        self._deque.append(new_data)
        self.workspace.append(
            new_data.s, new_data.y, new_data.ss, new_data.ys, new_data.yy
        )

    def get_last(self) -> IterationData:
        if not self._deque:
            raise RuntimeError("Memory is empty.")
        return self._deque[-1]

    def pop_last(self) -> IterationData:
        """Remove the newest pair and keep the shared workspace synchronized."""
        item = self._deque.pop()
        self.workspace.rebuild(iter(self._deque))
        return item

    def zero_memory_direction(self, g: np.ndarray, mu: np.float64) -> np.ndarray:
        return -g / (self.zero_memory_hessian_scale(mu) + mu)

    def zero_memory_hessian_scale(self, mu: np.float64) -> np.float64:
        if mu > 0.0:
            return self.zero_regularized_hessian_scale
        return self.zero_hessian_scale
