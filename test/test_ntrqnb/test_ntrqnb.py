import numpy as np

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn import qn
from qnlab.solver.qn_ntrqn import qn_ntrqn
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
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    method = Method("NTRQNB", "cautious", "damped", "bfgs")
    parameter = NTRQNParameter(2, {"gtol": np.float64(1e-9), "max_iterations": 100})
    callback = Callback(save_xs=True)

    code, fx, x = qn_ntrqnb(problem, bounds, parameter, method, callback)

    assert code == RetCode.SUCCESS
    np.testing.assert_allclose(x, [1.0, -1.0], atol=1e-8)
    assert fx == np.float64(1.0)
    assert all(np.all((-1.0 <= iterate) & (iterate <= 1.0)) for iterate in callback.xs)
    assert callback.gnorms[-1] <= parameter.gtol


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


def test_ntrqnb_matches_ntrqn_when_bounds_are_inactive():
    parameter = NTRQNParameter(2, {"gtol": np.float64(1e-9)})
    method = Method("NTRQN", "cautious", "damped", "bfgs")

    unconstrained = QuadraticProblem([10.0, 0.5])
    unconstrained_callback = Callback(save_xs=True)
    unconstrained_result = qn_ntrqn(
        unconstrained, parameter, method, unconstrained_callback
    )

    boxed = QuadraticProblem([10.0, 0.5])
    boxed_callback = Callback(save_xs=True)
    boxed_result = qn_ntrqnb(
        boxed,
        [(-100.0, 100.0), (-100.0, 100.0)],
        parameter,
        method,
        boxed_callback,
    )

    assert boxed_result[0] == unconstrained_result[0]
    np.testing.assert_allclose(boxed_result[1], unconstrained_result[1])
    np.testing.assert_allclose(boxed_result[2], unconstrained_result[2])
    np.testing.assert_allclose(boxed_callback.xs, unconstrained_callback.xs)


def test_scipy_boxed_callback_uses_projected_gradient():
    problem = QuadraticProblem([10.0, 0.5])
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    method = Method(base="SciPy", scipy_method="L-BFGS-B")
    callback = Callback(save_xs=True)

    code, _, x = qn(problem, method, callback=callback, bounds=bounds)

    assert code == RetCode.SUCCESS
    np.testing.assert_allclose(x, [1.0, -1.0])
    assert callback.gnorms[-1] == 0.0
