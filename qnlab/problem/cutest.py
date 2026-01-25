import numpy as np
import numpy.typing as npt

from qnlab.problem.base import BaseProblem


class CUTEstQNProblem(BaseProblem):
    def __init__(
        self,
        problem_name: str,
        precision: int = 64,
        noise: np.float64 = np.float64(0.0),
    ) -> None:
        import pycutest

        # Validate precision parameter
        if precision not in [64, 32, 16, -1]:
            raise ValueError(
                f"Precision must be one of [64, 32, 16, -1], got {precision}"
            )

        if noise < 0.0:
            raise ValueError("Noise level must be non-negative.")

        self.precision = precision
        self.noise = noise
        self.prob = pycutest.import_problem(problem_name)
        super().__init__(problem_name, self.prob.n, self.prob.x0.astype(np.float64))

    def _f(self, x: npt.NDArray[np.float64]) -> np.float64:
        if self.noise > 0.0:
            f = np.float64(self.prob.obj(x))  # type: ignore
            return f + np.random.uniform(-self.noise, self.noise)
        if self.precision == -1:
            # Return 0 for f when precision is -1
            return np.float64(0.0)
        elif self.precision == 64:
            return np.float64(self.prob.obj(x))  # type: ignore
        else:
            # For 32-bit or 16-bit precision
            x_low = self._convert_to_precision_array(x)
            result_raw = self.prob.obj(x_low)
            result = np.float64(result_raw)  # type: ignore
            return self._convert_to_precision_scalar(result)

    def _g(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.noise > 0.0:
            g = np.array(self.prob.grad(x), dtype=np.float64)  # type: ignore
            return g + np.random.uniform(-self.noise, self.noise, size=g.shape)
        if self.precision == -1 or self.precision == 64:
            # For -1 precision, compute gradient normally in 64-bit
            # For the constrained problems,
            # prob.grad is not the same as prob.lagjac(x)[0].
            return np.array(self.prob.lagjac(x)[0], dtype=np.float64)
        else:
            # For 32-bit or 16-bit precision
            x_low = self._convert_to_precision_array(x)
            result = np.array(self.prob.lagjac(x_low)[0], dtype=np.float64)
            return self._convert_to_precision_array(result)

    def _hessian(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.noise > 0.0:
            h = np.array(self.prob.hess(x), dtype=np.float64)  # type: ignore
            noise_matrix = np.random.uniform(
                -self.noise,
                self.noise,
                size=h.shape,
            )
            return h + noise_matrix
        if self.precision == 64 or self.precision == -1:
            return np.array(self.prob.hess(x), dtype=np.float64)
        else:
            # For 32-bit or 16-bit precision
            x_low = self._convert_to_precision_array(x)
            result = np.array(self.prob.hess(x_low), dtype=np.float64)
            return self._convert_to_precision_array(result)

    def _convert_to_precision_array(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Convert array to specified precision and back to 64-bit."""
        if self.precision == 32:
            return x.astype(np.float32).astype(np.float64)
        elif self.precision == 16:
            return x.astype(np.float16).astype(np.float64)
        else:
            return x

    def _convert_to_precision_scalar(self, x: np.float64) -> np.float64:
        """Convert scalar to specified precision and back to 64-bit."""
        if self.precision == 32:
            return np.float64(x.astype(np.float32))
        elif self.precision == 16:
            return np.float64(x.astype(np.float16))
        else:
            return x

    def get_eps(self) -> np.float64:
        eps = np.float64(0.0)
        if self.precision == 64:
            eps = np.float64(np.finfo(np.float64).eps) * 1e7 + self.noise
        elif self.precision == 32:
            eps = np.float64(np.finfo(np.float32).eps) * 1e4 + self.noise
        elif self.precision == 16:
            eps = np.float64(np.finfo(np.float16).eps) * 1e1 + self.noise
        elif self.precision == -1:
            eps = np.float64(1.0 - 1e-10)
        else:
            raise ValueError("Invalid precision value.")
        return np.clip(eps, 0, np.float64(1.0 - 1e-10))

    def get_noise(self) -> np.float64:
        """Returns the noise level for this problem."""
        return self.noise


if __name__ == "__main__":
    print(np.float64(np.finfo(np.float16).eps) * 1e2)
