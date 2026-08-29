import numpy as np

from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.solver.qn import qn
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def test_offo_uses_only_gradient_evaluations() -> None:
    problem = RosenbrockProblem(2)
    callback = Callback()

    qn(
        problem,
        Method("OFFO", label="OFFO"),
        {"max_iterations": 5, "gtol": np.float64(0.0)},
        callback,
    )

    assert problem.call_f == 0
    assert problem.call_g == 6
    assert np.all(np.isnan(callback.fxs))
    assert callback.calls == [0, 2, 3, 4, 5, 6]
