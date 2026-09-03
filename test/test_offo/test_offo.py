import numpy as np

from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def test_astr1_adagrad_step_and_oracle_accounting() -> None:
    problem = RosenbrockProblem(2)
    callback = Callback(save_xs=True)
    x0 = np.copy(problem.x0)
    g0 = problem.g(x0, count=False)

    qn(
        problem,
        Method("OFFO", label="ASTR1-Adagrad"),
        {"max_iterations": 5, "gtol": np.float64(0.0)},
        callback,
    )

    assert problem.call_f == 0
    assert problem.call_g == 6
    assert not np.isnan(callback.gnorms[0])
    assert np.all(np.isnan(callback.fxs[1:]))
    assert callback.calls == [0, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(
        callback.xs[1], x0 - g0 / np.sqrt(np.float64(1e-2) + np.square(g0))
    )


if __name__ == "__main__":
    test_astr1_adagrad_step_and_oracle_accounting()
