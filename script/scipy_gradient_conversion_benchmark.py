"""Benchmark SciPy L-BFGS-B gradient conversion strategies.

This module does not modify the installed SciPy package.  It obtains the source
of SciPy's private ``_minimize_lbfgsb`` Python driver and compiles copies in
which exactly one gradient-conversion line differs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import scipy.optimize._lbfgsb_py as _lbfgsb_py

from qnlab.parameter import LineParameter, NTRQNParameter
from qnlab.problem.ill_quadratic import IllQuadraticProblem
from qnlab.solver.qn_line import qn_line
from qnlab.solver.qn_ntrqn import qn_ntrqn
from qnlab.util.method import Method


CONVERSION_LINES = {
    "astype": "g = g.astype(np.float64)",
    "asarray": "g = np.asarray(g, dtype=np.float64)",
    "asarray_order_c": 'g = np.asarray(g, dtype=np.float64, order="C")',
}


def make_minimize_lbfgsb(conversion_line: str) -> Callable:
    """Return a copy of SciPy's driver with only its conversion line changed."""
    source = inspect.getsource(_lbfgsb_py._minimize_lbfgsb)
    old_line = "g = g.astype(np.float64)"
    if source.count(old_line) != 1:
        raise RuntimeError(
            "Expected exactly one gradient conversion in "
            f"{inspect.getsourcefile(_lbfgsb_py._minimize_lbfgsb)}; "
            "the installed SciPy source has changed."
        )

    patched_source = source.replace(old_line, conversion_line)
    namespace = dict(vars(_lbfgsb_py))
    exec(compile(patched_source, "<lbfgsb-gradient-conversion>", "exec"), namespace)
    return namespace["_minimize_lbfgsb"]


def build_variants() -> dict[str, Callable]:
    """Build the baseline, proposed, and layout-safe comparison drivers."""
    return {
        name: make_minimize_lbfgsb(line)
        for name, line in CONVERSION_LINES.items()
    }


def conversion_semantics() -> list[dict[str, object]]:
    """Report copying and layout behavior for representative gradients."""
    contiguous64 = np.ones(16, dtype=np.float64)
    contiguous32 = np.ones(16, dtype=np.float32)
    noncontiguous64 = np.ones(32, dtype=np.float64)[::2]
    inputs = {
        "contiguous_float64": contiguous64,
        "contiguous_float32": contiguous32,
        "noncontiguous_float64": noncontiguous64,
    }
    converters = {
        "astype": lambda g: g.astype(np.float64),
        "asarray": lambda g: np.asarray(g, dtype=np.float64),
        "asarray_order_c": lambda g: np.asarray(g, dtype=np.float64, order="C"),
    }

    rows = []
    for input_name, gradient in inputs.items():
        for variant, convert in converters.items():
            converted = convert(gradient)
            rows.append(
                {
                    "input": input_name,
                    "variant": variant,
                    "same_object": converted is gradient,
                    "shares_memory": np.shares_memory(converted, gradient),
                    "dtype": str(converted.dtype),
                    "c_contiguous": converted.flags.c_contiguous,
                }
            )
    return rows


def benchmark_conversion(
    dimensions: tuple[int, ...],
    repeats: int = 5,
    target_bytes_per_sample: int = 256 * 1024**2,
) -> list[dict[str, object]]:
    """Time conversion alone, using enough calls to move about target_bytes."""
    converters = {
        "astype": lambda g: g.astype(np.float64),
        "asarray": lambda g: np.asarray(g, dtype=np.float64),
        "asarray_order_c": lambda g: np.asarray(g, dtype=np.float64, order="C"),
    }
    rows = []
    for dtype in (np.float64, np.float32):
        for n in dimensions:
            gradient = np.ones(n, dtype=dtype)
            calls = max(10, min(100_000, target_bytes_per_sample // (8 * n)))
            for sample in range(repeats):
                order = list(converters)
                random.Random(10_000 * sample + n + np.dtype(dtype).itemsize).shuffle(order)
                for variant in order:
                    convert = converters[variant]
                    gc.collect()
                    start = time.perf_counter()
                    converted = None
                    for _ in range(calls):
                        converted = convert(gradient)
                    elapsed = time.perf_counter() - start
                    assert converted is not None
                    rows.append(
                        {
                            "dtype": np.dtype(dtype).name,
                            "dimension": n,
                            "variant": variant,
                            "sample": sample,
                            "calls": calls,
                            "elapsed_seconds": elapsed,
                            "nanoseconds_per_call": elapsed * 1e9 / calls,
                        }
                    )
    return rows


@dataclass
class SolverTiming:
    solver: str
    dimension: int
    requested_iterations: int
    repeat: int
    elapsed_seconds: float
    iterations: int | None
    function_calls: int
    gradient_calls: int
    final_objective: float
    exit_status: str


def run_solver_once(
    solver: str,
    scipy_variants: dict[str, Callable],
    dimension: int,
    iterations: int,
    repeat: int,
) -> SolverTiming:
    problem = IllQuadraticProblem(dimension)
    evaluation_limit = 50 * iterations + 100

    start = time.perf_counter()
    if solver in scipy_variants:
        result = scipy_variants[solver](
            problem.f,
            problem.x0.copy(),
            jac=problem.g,
            bounds=None,
            maxcor=10,
            ftol=0.0,
            gtol=0.0,
            maxfun=evaluation_limit,
            maxiter=iterations,
            maxls=40,
        )
        elapsed = time.perf_counter() - start
        completed_iterations = int(result.nit)
        final_objective = float(result.fun)
        exit_status = str(result.message)
    elif solver == "qnlab qn_line":
        method = Method(base="Line", store="raw", secant="raw", update="bfgs")
        parameter = LineParameter(
            dimension,
            {
                "m": 10,
                "max_iterations": iterations,
                "max_evaluations": evaluation_limit,
                "past": 0,
                "ftol": 0.0,
                "gtol": 0.0,
            },
        )
        code, final_objective, _ = qn_line(
            problem, parameter, method, callback=None
        )
        elapsed = time.perf_counter() - start
        completed_iterations = (
            iterations if "MAXIMUMITERATION" in str(code) else None
        )
        exit_status = str(code)
    elif solver == "qnlab qn_ntrqn":
        method = Method(
            base="NTRQN", store="cautious", secant="damped", update="bfgs"
        )
        parameter = NTRQNParameter(
            dimension,
            {
                "m": 10,
                "max_iterations": iterations,
                "max_evaluations": evaluation_limit,
                "past": 0,
                "ftol": 0.0,
                "gtol": 0.0,
            },
        )
        code, final_objective, _ = qn_ntrqn(
            problem, parameter, method, callback=None, verbose=False
        )
        elapsed = time.perf_counter() - start
        completed_iterations = (
            iterations if "MAXIMUMITERATION" in str(code) else None
        )
        exit_status = str(code)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return SolverTiming(
        solver=solver,
        dimension=dimension,
        requested_iterations=iterations,
        repeat=repeat,
        elapsed_seconds=elapsed,
        iterations=completed_iterations,
        function_calls=problem.call_f,
        gradient_calls=problem.call_g,
        final_objective=float(final_objective),
        exit_status=exit_status,
    )


def benchmark_solver(
    dimensions: tuple[int, ...],
    iterations: int = 1_000,
    repeats: int = 5,
) -> list[dict[str, object]]:
    """Benchmark complete L-BFGS-B runs in randomized within-repeat order."""
    scipy_variants = {
        f"SciPy {name}": driver for name, driver in build_variants().items()
    }
    solvers = (*scipy_variants, "qnlab qn_line", "qnlab qn_ntrqn")
    for solver in solvers:
        run_solver_once(solver, scipy_variants, 200, 5, repeat=-1)

    cases = [(solver, n) for n in dimensions for solver in solvers]
    rows = []
    for repeat in range(repeats):
        shuffled = cases.copy()
        random.Random(20260726 + repeat).shuffle(shuffled)
        for solver, n in shuffled:
            timing = run_solver_once(
                solver, scipy_variants, n, iterations, repeat
            )
            rows.append(asdict(timing))
    return rows


def check_noncontiguous_gradient() -> list[dict[str, object]]:
    """Check whether each driver accepts a strided float64 user gradient."""
    variants = build_variants()
    rows = []
    for name, driver in variants.items():
        diagonal = np.linspace(1.0, 200.0, 200)

        def fun(x: np.ndarray) -> np.float64:
            return np.float64(0.5 * np.dot(x * diagonal, x))

        def jac(x: np.ndarray) -> np.ndarray:
            contiguous = diagonal * x
            storage = np.empty(2 * x.size, dtype=np.float64)
            storage[::2] = contiguous
            gradient = storage[::2]
            assert not gradient.flags.c_contiguous
            return gradient

        try:
            result = driver(
                fun,
                np.ones(200),
                jac=jac,
                bounds=None,
                maxcor=10,
                ftol=0.0,
                gtol=0.0,
                maxfun=250,
                maxiter=5,
                maxls=20,
            )
        except Exception as error:
            rows.append(
                {
                    "variant": name,
                    "accepted": False,
                    "iterations": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        else:
            rows.append(
                {
                    "variant": name,
                    "accepted": True,
                    "iterations": int(result.nit),
                    "error": "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "SciPy",
        help="Directory for CSV and environment metadata.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()

    conversion_dimensions = (1_000, 10_000, 100_000, 1_000_000)
    solver_dimensions = (1_000, 3_000, 10_000, 30_000, 100_000, 300_000)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        args.output_dir / "lbfgsb_gradient_conversion_semantics.csv",
        conversion_semantics(),
    )
    write_csv(
        args.output_dir / "lbfgsb_gradient_conversion_microbenchmark.csv",
        benchmark_conversion(conversion_dimensions, repeats=args.repeats),
    )
    write_csv(
        args.output_dir / "lbfgsb_gradient_conversion_solver.csv",
        benchmark_solver(
            solver_dimensions,
            iterations=args.iterations,
            repeats=args.repeats,
        ),
    )
    write_csv(
        args.output_dir / "lbfgsb_gradient_conversion_layout.csv",
        check_noncontiguous_gradient(),
    )

    environment = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "scipy_source": inspect.getsourcefile(_lbfgsb_py._minimize_lbfgsb),
    }
    (args.output_dir / "lbfgsb_gradient_conversion_environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote benchmark data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
