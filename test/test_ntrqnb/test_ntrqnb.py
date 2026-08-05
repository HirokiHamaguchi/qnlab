import numpy as np

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn import qn
from qnlab.solver.qn_ntrqn import qn_ntrqn
from qnlab.solver.qn_ntrqnb import (
    _CompactBFGS,
    _generalized_cauchy_point,
    _subspace_minimization,
    max_feasible_step,
    projected_gradient,
    qn_ntrqnb,
)
from qnlab.update.update import get_direction_reg
from qnlab.util.callback import Callback
from qnlab.util.memory_interface import QuasiNewtonMemory
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


class BoundaryScaledProblem(BaseProblem):
    """A large active-bound gradient must not shrink the free-variable step."""

    def __init__(self):
        super().__init__("boundary scaled", 2, np.zeros(2, dtype=np.float64))

    def _f(self, x):
        return np.float64(1e12 * x[0] + 0.5 * x[0] ** 2 + 0.5 * (x[1] + 1) ** 2)

    def _g(self, x):
        return np.array([1e12 + x[0], x[1] + 1.0], dtype=np.float64)


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


def test_box_direction_solves_coupled_quadratic_model():
    # B = [[10, 9], [9, 10]]. Clipping its unconstrained minimizer [1, -1]
    # would give [0.2, -1], whereas the box-QP minimizer is [0.2, -0.28].
    operator = _CompactBFGS(
        diagonal=np.float64(1.0),
        vectors=np.array([[1.0], [1.0]]),
        coefficients=np.array([9.0]),
    )
    x = np.zeros(2)
    g = np.array([-1.0, 1.0])
    lb = np.full(2, -np.inf)
    ub = np.array([0.2, np.inf])

    cauchy, active = _generalized_cauchy_point(x, g, lb, ub, operator)
    candidate = _subspace_minimization(cauchy, active, x, g, lb, ub, operator)

    np.testing.assert_allclose(candidate, [0.2, -0.28], atol=1e-14)


def test_compact_regularization_matches_ntrqn_two_loop():
    method = Method("NTRQNB", "raw", "raw", "bfgs")
    q = np.array([[4.0, 0.5, 0.0], [0.5, 3.0, 0.25], [0.0, 0.25, 2.0]])
    b = np.array([1.0, -2.0, 0.5])
    points = [
        np.zeros(3),
        np.array([0.2, -0.1, 0.3]),
        np.array([0.1, 0.25, 0.4]),
    ]

    def f(point):
        return np.float64(0.5 * point @ q @ point + b @ point)

    def grad(point):
        return q @ point + b

    memory = QuasiNewtonMemory(grad(points[0]), 10, method)
    for previous, current in zip(points, points[1:]):
        memory.add_new_data(
            current,
            f(current),
            grad(current),
            previous,
            f(previous),
            grad(previous),
            None,
            np.float64(0.0),
        )

    mu = np.float64(0.7)
    g = np.array([-0.4, 0.3, 1.2])
    operator = _CompactBFGS.from_memory(memory, mu, 3)
    matrix = np.column_stack([operator.apply(np.eye(3)[i]) for i in range(3)])
    expected = get_direction_reg(method, np.zeros(3), g, memory, mu)

    np.testing.assert_allclose(np.linalg.solve(matrix, -g), expected, rtol=1e-12)


def test_active_gradient_does_not_shrink_initial_free_step():
    problem = BoundaryScaledProblem()
    method = Method("NTRQNB", "raw", "raw", "bfgs")
    parameter = NTRQNParameter(2, {"gtol": np.float64(1e-10)})

    code, fx, x = qn_ntrqnb(
        problem,
        [(0.0, None), (None, None)],
        parameter,
        method,
    )

    assert code == RetCode.SUCCESS
    np.testing.assert_allclose(x, [0.0, -1.0], atol=1e-12)
    assert fx == 0.0


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
