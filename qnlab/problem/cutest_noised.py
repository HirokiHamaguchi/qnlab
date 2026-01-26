import numpy as np
import numpy.typing as npt

from qnlab.problem.cutest import CUTEstQNProblem


class CUTEstNoisedProblem(CUTEstQNProblem):
    def __init__(
        self,
        problem_name: str,
        precision: int = 64,
        noise: np.float64 = np.float64(0.0),
    ) -> None:
        super().__init__(problem_name, precision)
        if noise < 0.0:
            raise ValueError("Noise level must be non-negative.")
        self.noise = np.float64(noise)
        self.rng = np.random.default_rng(seed=0)

    def _f(self, x: npt.NDArray[np.float64]) -> np.float64:
        f = super()._f(x)
        return f + self.rng.uniform(-self.noise, self.noise)

    def _g(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        g = super()._g(x)
        return g + self.rng.uniform(-self.noise, self.noise, size=g.shape)

    def _hessian(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        h = super()._hessian(x)
        noise_matrix = self.rng.uniform(-self.noise, self.noise, size=h.shape)
        return h + noise_matrix

    def get_machine_eps(self) -> np.float64:
        eps = np.float64(super().get_machine_eps() + self.noise)
        return np.clip(eps, 0, np.float64(1.0 - 1e-10))

    def get_noise(self) -> np.float64:
        return self.noise
