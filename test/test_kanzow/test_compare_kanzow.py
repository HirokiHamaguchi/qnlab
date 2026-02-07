import importlib
import os
import sys
from typing import Union

import numpy as np

from qnlab.parameter import KanzowParameter
from qnlab.problem.dixon_price import DixonPriceProblem
from qnlab.problem.powell import PowellProblem
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.problem.zakharov import ZakharovProblem
from qnlab.solver.qn_kanzow import qn_kanzow
from qnlab.util.method import Method

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../submodules",
        "paper-regularized-qn-benchmark",
    )
)
kanzow = importlib.import_module("regLBFGS")
kanzow_sec = importlib.import_module("regLBFGSsec")
utility = importlib.import_module("utility")


def trial(
    prob: Union[RosenbrockProblem, PowellProblem, DixonPriceProblem, ZakharovProblem],
    maxIter: int,
):
    name = prob.__class__.__name__.replace("Problem", "")
    print(f"----- Problem name: {name} -----")

    param = KanzowParameter(
        prob.n, {"m": 5, "gtol": np.float64(1e-4), "max_iterations": maxIter}
    )
    info, fx, x_opt = qn_kanzow(prob, param, method=Method("Kanzow"))

    utility.parameters.maxIter = maxIter
    x_true, _ = kanzow.solveNonmonotone(prob.f, prob.g, prob.x0)
    fx_true = prob.f(x_true)

    print(fx, fx_true)
    assert fx == fx_true

    if isinstance(prob, RosenbrockProblem):
        assert np.allclose(x_opt, x_true, atol=1e-5), f"{x_opt=} != {x_true=}"

    print("Function value and solution comparison passed!")


def trialSec(
    prob: Union[RosenbrockProblem, PowellProblem, DixonPriceProblem, ZakharovProblem],
    maxIter: int,
):
    name = prob.__class__.__name__.replace("Problem", "")
    print(f"----- Problem name: {name} -----")

    param = KanzowParameter(
        prob.n, {"m": 5, "gtol": np.float64(1e-4), "max_iterations": maxIter}
    )
    info, fx, x_opt = qn_kanzow(prob, param, method=Method("KanzowSec"))

    utility.parameters.maxIter = maxIter
    x_true, _ = kanzow_sec.solveNonmonotone(prob.f, prob.g, prob.x0)
    fx_true = prob.f(x_true)

    print(fx, fx_true)
    assert fx == fx_true

    if isinstance(prob, RosenbrockProblem):
        assert np.allclose(x_opt, x_true, atol=1e-5), f"{x_opt=} != {x_true=}"

    print("Function value and solution comparison passed!")


def test_compare_kanzow():
    for maxIter in [5, 10, 15, 50]:
        prob = RosenbrockProblem(n=5)
        trial(prob, maxIter)
        trialSec(prob, maxIter)

        prob = PowellProblem()
        trial(prob, maxIter)
        trialSec(prob, maxIter)

        prob = DixonPriceProblem()
        trial(prob, maxIter)
        trialSec(prob, maxIter)

        prob = ZakharovProblem()
        trial(prob, maxIter)
        trialSec(prob, maxIter)


if __name__ == "__main__":
    test_compare_kanzow()
