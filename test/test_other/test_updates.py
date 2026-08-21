import numpy as np
import numpy.typing as npt

from qnlab.update.update import check_direction, get_direction, get_direction_reg
from qnlab.util.iteration_data import (
    CAUTIOUS_CURVATURE_LOWER,
    CAUTIOUS_CURVATURE_UPPER,
    IterationData,
)
from qnlab.util.memory_interface import LBFGSWorkspace, QuasiNewtonMemory
from qnlab.util.method import Method


def generate_lm(
    n, maxlen=3
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    QuasiNewtonMemory,
]:
    lm = QuasiNewtonMemory(np.zeros(n), maxlen=maxlen, method=Method())

    A = np.random.randn(n, n)
    A = A @ A.T  # SPD
    b = np.random.randn(n)
    c = np.random.randn()

    def f(x) -> np.float64:
        return 0.5 * x @ A @ x + b @ x + c

    xp = np.random.randn(n)
    gp = np.random.randn(n)

    while True:
        # generate a nonzero s vector
        xk = np.random.randn(n)
        gk = np.random.randn(n)

        lm.add_new_data(xk, f(xk), gk, xp, f(xp), gp, None, np.float64(0.0))

        if len(lm) and (
            np.abs(lm.get_last().ys) < 1e-3 or np.abs(lm.get_last().yy) < 1e-3
        ):
            lm.pop_last()
            continue

        if len(lm) >= maxlen:
            return xk, gk, lm


def test_dir_BFGS_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "bfgs")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_lbfgs = get_direction(method, xk, gk, lm)
        check_direction(method, n, gk, d_lbfgs, lm)
    print("BFGS passed")


def test_lbfgs_workspace_keeps_contiguous_pairs_and_gram_matrices():
    workspace = LBFGSWorkspace(n=5, capacity=3)
    rng = np.random.default_rng(12)
    pairs = [(rng.normal(size=5), rng.normal(size=5)) for _ in range(4)]
    for step, gradient in pairs:
        workspace.append(step, gradient)

    expected_steps = np.column_stack([pair[0] for pair in pairs[-3:]])
    expected_gradients = np.column_stack([pair[1] for pair in pairs[-3:]])
    np.testing.assert_allclose(workspace.steps, expected_steps)
    np.testing.assert_allclose(workspace.gradients, expected_gradients)
    assert workspace._steps.flags.f_contiguous
    assert workspace._gradients.flags.f_contiguous
    np.testing.assert_allclose(
        workspace.step_products, expected_steps.T @ expected_steps
    )
    np.testing.assert_allclose(
        workspace.step_gradient, expected_steps.T @ expected_gradients
    )
    np.testing.assert_allclose(
        workspace.gradient_products, expected_gradients.T @ expected_gradients
    )


def test_cautious_rule_enforces_both_uniform_curvature_bounds():
    method = Method("NTRQN", "cautious", "raw", "bfgs")
    accepted = IterationData()
    is_valid, _ = accepted.set(
        np.array([1.0]),
        np.float64(0.0),
        np.array([1.0]),
        np.array([0.0]),
        np.float64(0.0),
        np.array([0.0]),
        method,
        np.float64(0.0),
    )
    assert is_valid
    assert accepted.ys >= CAUTIOUS_CURVATURE_LOWER * accepted.ss
    assert accepted.ys >= accepted.yy / CAUTIOUS_CURVATURE_UPPER

    rejected = IterationData()
    is_valid, message = rejected.set(
        np.array([1000.0]),
        np.float64(0.0),
        np.array([1e-6]),
        np.array([0.0]),
        np.float64(0.0),
        np.array([0.0]),
        method,
        np.float64(0.0),
    )
    assert not is_valid
    assert message == "skip by cautious update"


def test_zero_memory_direction_uses_positive_fixed_hessian_scale():
    gradient = np.array([3.0, 4.0])
    memory = QuasiNewtonMemory(
        gradient,
        maxlen=3,
        method=Method("NTRQN", "cautious", "damped", "bfgs"),
    )

    np.testing.assert_allclose(
        memory.zero_memory_direction(gradient, np.float64(0.0)),
        -gradient / 5.0,
    )
    np.testing.assert_allclose(
        memory.zero_memory_direction(gradient, np.float64(2.0)),
        -gradient / 7.0,
    )


def test_scalar_damping_matches_documented_rule():
    data = IterationData()
    is_valid, message = data.set(
        np.array([1.0]),
        np.float64(0.0),
        np.array([-1.0]),
        np.array([0.0]),
        np.float64(0.0),
        np.array([0.0]),
        Method("NTRQN", "raw", "damped", "bfgs"),
        np.float64(0.0),
    )

    assert is_valid, message
    np.testing.assert_allclose(data.y, np.array([0.2]))
    np.testing.assert_allclose(data.ys, np.float64(0.2))


def test_workspace_two_loop_matches_pairwise_reference():
    rng = np.random.default_rng(21)
    n = 8
    matrix = rng.normal(size=(n, n))
    matrix = matrix.T @ matrix + np.eye(n)
    method = Method("NTRQN", "raw", "raw", "bfgs")
    point = np.zeros(n)
    memory = QuasiNewtonMemory(matrix @ point, maxlen=4, method=method)
    for _ in range(6):
        new_point = point + rng.normal(size=n)
        memory.add_new_data(
            new_point,
            np.float64(0.0),
            matrix @ new_point,
            point,
            np.float64(0.0),
            matrix @ point,
            None,
            np.float64(0.0),
        )
        point = new_point

    gradient = rng.normal(size=n)
    for mu in (np.float64(0.0), np.float64(0.7)):
        direction = -gradient.copy()
        alphas = np.empty(len(memory))
        items = list(memory)
        for index in range(len(items) - 1, -1, -1):
            item = items[index]
            denominator = item.ys + mu * item.ss
            alphas[index] = np.dot(item.s, direction) / denominator
            direction -= alphas[index] * (item.y + mu * item.s)
        last = items[-1]
        direction *= (last.ys + mu * last.ss) / (
            last.yy + 2.0 * mu * last.ys + mu * mu * last.ss
        )
        for index, item in enumerate(items):
            beta = np.dot(item.y + mu * item.s, direction) / (item.ys + mu * item.ss)
            direction += (alphas[index] - beta) * item.s

        np.testing.assert_allclose(
            get_direction_reg(method, point, gradient, memory, mu), direction
        )


def test_dir_DFP_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "dfp")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_dfp = get_direction(method, xk, gk, lm)
        check_direction(method, n, gk, d_dfp, lm)
    print("DFP passed")


def test_dir_SR1_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "sr1")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_sr1 = get_direction(method, xk, gk, lm)
        check_direction(method, n, gk, d_sr1, lm)
    print("SR1 passed")


def test_dir_PSB_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "psb")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_psb = get_direction(method, xk, gk, lm)
        check_direction(method, n, gk, d_psb, lm)
    print("PSB passed")


if __name__ == "__main__":
    np.random.seed(0)  # Since numerically instable, we set a seed for reproducibility
    test_dir_BFGS_and_check()
    test_dir_DFP_and_check()
    test_dir_SR1_and_check()
    test_dir_PSB_and_check()
