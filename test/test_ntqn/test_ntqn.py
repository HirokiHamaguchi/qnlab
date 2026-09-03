import numpy as np

import qnlab.solver.qn_ntqn as qn_ntqn_module
from qnlab.parameter import NtqnParameter
from qnlab.problem.dixon_price import DixonPriceProblem
from qnlab.problem.powell import PowellProblem
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.problem.zakharov import ZakharovProblem
from qnlab.solver.qn_ntqn import qn_ntqn
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def trial(
    prob: RosenbrockProblem | PowellProblem | DixonPriceProblem | ZakharovProblem,
):
    """
    Compare qnlab wrapper with direct ntqn call for final objective value and solution.
    """
    name = prob.__class__.__name__.replace("Problem", "")
    print(f"----- Problem name: {name} -----")

    param = NtqnParameter(prob.n)
    info, fx, _x_opt = qn_ntqn(prob, param, method=Method("NTQN"))
    print(f"Info (qnlab): {info}")
    print(f"fx (qnlab): {fx}")


def test_rosenbrock():
    prob = RosenbrockProblem(n=5)
    trial(prob)


def test_powell():
    prob = PowellProblem()
    trial(prob)


def test_dixon_price():
    prob = DixonPriceProblem()
    trial(prob)


def test_zakharov():
    prob = ZakharovProblem()
    trial(prob)


def test_external_ntqn_final_iterate_is_recorded(monkeypatch):
    problem = RosenbrockProblem(n=2)
    final_x = np.ones(2)

    def fake_bfgs_e(*args, **kwargs):
        return final_x, np.float64(0.0), 1, 1, 1, 0, {}

    monkeypatch.setattr(qn_ntqn_module.ntqn, "bfgs_e", fake_bfgs_e)
    callback = Callback()
    parameter = NtqnParameter(2, {"terminate": 3, "stop_at_gtol": 0})

    qn_ntqn(problem, parameter, Method("NTQN"), callback)

    assert len(callback.gnorms) == 2
    assert callback.gnorms[-1] == 0.0
    assert callback.others["NTQN termination flag"] == 0


if __name__ == "__main__":
    test_rosenbrock()
    test_powell()
    test_dixon_price()
    test_zakharov()
