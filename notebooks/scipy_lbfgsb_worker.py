"""Run SciPy L-BFGS-B timings in an isolated Python environment."""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_info, threadpool_limits


class ZeroChainQuadratic:
    def __init__(self, dimension: int):
        self.x0 = np.zeros(dimension)
        self.call_f = 0
        self.call_g = 0

    def f(self, x: np.ndarray) -> np.float64:
        self.call_f += 1
        differences = x[:-1] - x[1:]
        return np.float64(0.5 * x[0] ** 2 - x[0] + 0.5 * differences @ differences)

    def g(self, x: np.ndarray) -> np.ndarray:
        self.call_g += 1
        differences = x[:-1] - x[1:]
        gradient = np.zeros_like(x)
        gradient[0] = x[0] - 1.0
        gradient[:-1] += differences
        gradient[1:] -= differences
        return gradient


class IllQuadratic:
    def __init__(self, dimension: int):
        self.x0 = np.ones(dimension)
        self.weights = np.arange(1, dimension + 1, dtype=float)
        self.call_f = 0
        self.call_g = 0

    def f(self, x: np.ndarray) -> np.float64:
        self.call_f += 1
        return np.float64(0.5 * np.dot(self.weights * x, x))

    def g(self, x: np.ndarray) -> np.ndarray:
        self.call_g += 1
        return self.weights * x


PROBLEMS = {
    "zero_chain": ZeroChainQuadratic,
    "ill_quadratic": IllQuadratic,
}


@dataclass
class TimingResult:
    dimension: int
    requested_iterations: int
    repeat: int
    elapsed_seconds: float
    iterations: int
    function_calls: int
    gradient_calls: int
    final_objective: float
    exit_status: str


def run_once(
    problem_name: str,
    dimension: int,
    max_iterations: int,
    repeat: int,
    memory: int,
) -> TimingResult:
    problem = PROBLEMS[problem_name](dimension)
    evaluation_limit = 50 * max_iterations + 100
    start = time.perf_counter()
    result = minimize(
        problem.f,
        problem.x0.copy(),
        jac=problem.g,
        method="L-BFGS-B",
        bounds=None,
        callback=None,
        options={
            "maxcor": memory,
            "maxiter": max_iterations,
            "maxfun": evaluation_limit,
            "ftol": 0.0,
            "gtol": 0.0,
            "maxls": 40,
        },
    )
    elapsed = time.perf_counter() - start
    return TimingResult(
        dimension=dimension,
        requested_iterations=max_iterations,
        repeat=repeat,
        elapsed_seconds=elapsed,
        iterations=int(result.nit),
        function_calls=problem.call_f,
        gradient_calls=problem.call_g,
        final_objective=float(result.fun),
        exit_status=str(result.message),
    )


def main() -> None:
    config = json.load(sys.stdin)
    blas_threads = config["blas_threads"]
    thread_context = (
        threadpool_limits(limits=blas_threads, user_api="blas")
        if blas_threads is not None
        else nullcontext()
    )
    rows = []
    with thread_context:
        run_once(config["problem"], 600, 5, -1, config["memory"])
        for repeat in range(config["repeats"]):
            for dimension, iterations in config["cases"]:
                measurement = run_once(
                    config["problem"],
                    dimension,
                    iterations,
                    repeat,
                    config["memory"],
                )
                rows.append(asdict(measurement))
                print(
                    f"repeat={repeat + 1}/{config['repeats']}, n={dimension}, "
                    f"k={iterations}: {measurement.elapsed_seconds:.3f} s",
                    file=sys.stderr,
                )

        pools = threadpool_info()

    try:
        from scipy.version import git_revision
    except ImportError:
        git_revision = "unknown"
    json.dump(
        {
            "environment": {
                "python": sys.version.replace("\n", " "),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scipy_git_revision": git_revision,
                "platform": platform.platform(),
                "threadpools": pools,
            },
            "rows": rows,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
