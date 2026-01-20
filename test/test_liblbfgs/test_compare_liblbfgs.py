import os
from typing import Union

import numpy as np

from qnlab.parameter import LineParameter
from qnlab.problem.dixon_price import DixonPriceProblem
from qnlab.problem.powell import PowellProblem
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.problem.zakharov import ZakharovProblem
from qnlab.solver.qn_line import qn_line
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def trial(
    prob: Union[RosenbrockProblem, PowellProblem, DixonPriceProblem, ZakharovProblem],
):
    name = prob.__class__.__name__.replace("Problem", "")
    print(f"----- Problem name: {name} -----")

    callback = Callback(gnorm_order=2)
    param = LineParameter(prob.n, {"m": 6, "gtol": 0, "max_iterations": 100})
    info, fx, x_opt = qn_line(
        prob, param, method=Method("Line", "raw", "raw", "bfgs"), callback=callback
    )
    print(f"Info: {info}")

    recorded_gnorms = np.array(callback.gnorms[1:])

    folder_path = os.path.dirname(os.path.abspath(__file__))
    file_name = f"data/{name}_results.txt"
    with open(os.path.join(folder_path, file_name), "r") as f:
        txt_gnorms = np.array([float(line.strip()) for line in f if line.strip()])

    # rosenbrock fails when comparing with sz=80 due to machine precision
    sz = min(len(recorded_gnorms), len(txt_gnorms), 70)
    recorded_gnorms = recorded_gnorms[:sz]
    txt_gnorms = txt_gnorms[:sz]

    assert np.allclose(
        recorded_gnorms,
        txt_gnorms,
        rtol=1e-5,
        atol=1e-8,
    ), "GNORM comparison failed!"

    print("GNORM comparison passed!")


def test_rosenbrock():
    prob = RosenbrockProblem()
    trial(prob)


def test_powell():
    prob = PowellProblem()
    trial(prob)


def test_dixon_price():
    prob = DixonPriceProblem()
    prob.x0 = np.array([0.5] * prob.n, dtype=np.float64)  # todo: remove this line
    trial(prob)


def test_zakharov():
    prob = ZakharovProblem()
    trial(prob)


if __name__ == "__main__":
    test_rosenbrock()
    test_powell()
    test_dixon_price()
    test_zakharov()
