from types import SimpleNamespace

import numpy as np

import qnlab.solver.qn_scipy as qn_scipy_module
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.solver.qn_scipy import qn_scipy
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def test_scipy_final_iterate_is_recorded(monkeypatch):
    problem = RosenbrockProblem(n=2)
    final_x = np.ones(2)

    def fake_minimize(*args, **kwargs):
        return SimpleNamespace(
            success=True,
            fun=np.float64(0.0),
            x=final_x,
            message="success",
        )

    monkeypatch.setattr(qn_scipy_module.scipy.optimize, "minimize", fake_minimize)
    callback = Callback()

    qn_scipy(
        problem,
        Method("SciPy", scipy_method="L-BFGS-B"),
        callback=callback,
    )

    assert len(callback.gnorms) == 2
    assert callback.gnorms[-1] == 0.0
