import copy

import numpy as np
from qnlab.problem.base import BaseProblem
from qnlab.problem.cutest import CUTEstQNProblem
from qnlab.problem.cutest_noised import CUTEstNoisedProblem


def _problem(monkeypatch) -> CUTEstNoisedProblem:
    monkeypatch.setattr(
        CUTEstQNProblem,
        "_f",
        lambda self, x: np.float64(np.dot(x, x)),
    )
    monkeypatch.setattr(
        CUTEstQNProblem,
        "_g",
        lambda self, x: 2.0 * x,
    )
    problem = CUTEstNoisedProblem.__new__(CUTEstNoisedProblem)
    BaseProblem.__init__(problem, "test", n=2, x0=np.ones(2))
    problem.function_noise = np.float64(1e-3)
    problem.gradient_noise = np.float64(1e-3)
    problem.assumed_function_error = np.float64(1e-2)
    problem.rng = np.random.default_rng(seed=7)
    return problem


def test_uncounted_evaluations_do_not_advance_noise_rng(monkeypatch) -> None:
    problem = _problem(monkeypatch)
    initial_state = copy.deepcopy(problem.rng.bit_generator.state)

    problem.f(problem.x0, count=False)
    problem.g(problem.x0, count=False)

    assert problem.rng.bit_generator.state == initial_state


def test_counted_evaluations_advance_noise_rng(monkeypatch) -> None:
    problem = _problem(monkeypatch)
    initial_state = copy.deepcopy(problem.rng.bit_generator.state)

    problem.f(problem.x0)

    assert problem.rng.bit_generator.state != initial_state
