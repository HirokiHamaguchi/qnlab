import numpy as np
import numpy.typing as npt

from qnlab.problem.cutest import CUTEstQNProblem


class CUTEstNoisedProblem(CUTEstQNProblem):
    def __init__(
        self,
        problem_name: str,
        precision: int = 64,
        noise: np.float64 = np.float64(0.0),
        function_noise: np.float64 | None = None,
        gradient_noise: np.float64 | None = None,
        assumed_function_error: np.float64 | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(problem_name, precision)
        self.function_noise = np.float64(
            noise if function_noise is None else function_noise
        )
        self.gradient_noise = np.float64(
            noise if gradient_noise is None else gradient_noise
        )
        if self.function_noise < 0.0 or self.gradient_noise < 0.0:
            raise ValueError("Noise levels must be non-negative.")
        if (
            assumed_function_error is not None
            and not 0.0 <= assumed_function_error < 1.0
        ):
            raise ValueError("assumed_function_error must belong to [0, 1).")
        self.assumed_function_error = (
            None
            if assumed_function_error is None
            else np.float64(assumed_function_error)
        )
        self.noise = max(self.function_noise, self.gradient_noise)
        self.rng = np.random.default_rng(seed=seed)

    def _f(self, x: npt.NDArray[np.float64]) -> np.float64:
        f = super()._f(x)
        return f + self.rng.uniform(-self.function_noise, self.function_noise)

    def _g(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        g = super()._g(x)
        return g + self.rng.uniform(
            -self.gradient_noise, self.gradient_noise, size=g.shape
        )

    def _hessian(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        h = super()._hessian(x)
        noise_matrix = self.rng.uniform(
            -self.gradient_noise, self.gradient_noise, size=h.shape
        )
        return h + noise_matrix

    def get_machine_eps(self) -> np.float64:
        if self.assumed_function_error is not None:
            return self.assumed_function_error
        eps = np.float64(super().get_machine_eps() + self.function_noise)
        return np.clip(eps, 0, np.float64(1.0 - 1e-10))

    def get_noise(self) -> npt.NDArray[np.float64]:
        return np.full(self.n, self.gradient_noise, dtype=np.float64)
