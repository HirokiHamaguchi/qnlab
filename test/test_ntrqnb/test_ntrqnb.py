import numpy as np
from scipy.optimize import Bounds

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn_ntrqnb import max_feasible_step, projected_gradient, qn_ntrqnb
from qnlab.util.callback import Callback
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


class QuadraticProblem(BaseProblem):
    def __init__(self, x0):
        super().__init__("bounded quadratic", len(x0), np.asarray(x0, dtype=np.float64))

    def _f(self, x):
        error = x - np.array([2.0, -2.0])
        return np.float64(0.5 * np.dot(error, error))

    def _g(self, x):
        return x - np.array([2.0, -2.0])


def test_projected_gradient_and_feasible_step():
    x = np.array([0.0, 1.0, 0.5])
    g = np.array([2.0, -3.0, 0.25])
    lb = np.zeros(3)
    ub = np.ones(3)
    np.testing.assert_allclose(projected_gradient(x, g, lb, ub), [0.0, 0.0, 0.25])
    assert max_feasible_step(x, np.array([1.0, -2.0, 0.0]), lb, ub) == 0.5


def test_ntrqnb_finds_boundary_solution():
    problem = QuadraticProblem([10.0, 0.5])
    bounds = Bounds([-1.0, -1.0], [1.0, 1.0])
    method = Method("NTRQNB", "cautious", "damped", "bfgs")
    parameter = NTRQNParameter(2, {"gtol": np.float64(1e-9), "max_iterations": 100})
    callback = Callback(save_xs=True)

    code, fx, x = qn_ntrqnb(problem, bounds, parameter, method, callback)

    assert code == RetCode.SUCCESS
    np.testing.assert_allclose(x, [1.0, -1.0], atol=1e-8)
    assert fx == np.float64(1.0)
    assert all(np.all((-1.0 <= iterate) & (iterate <= 1.0)) for iterate in callback.xs)


def test_ntrqnb_detects_initial_constrained_stationarity():
    problem = QuadraticProblem([1.0, -1.0])
    method = Method("NTRQNB", "raw", "raw", "bfgs")
    code, _, _ = qn_ntrqnb(
        problem,
        [(-1.0, 1.0), (-1.0, 1.0)],
        NTRQNParameter(2),
        method,
    )
    assert code == RetCode.ALREADY_MINIMIZED
