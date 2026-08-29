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
    max_sz: int,
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

    sz = min(len(recorded_gnorms), len(txt_gnorms), max_sz)
    recorded_gnorms = recorded_gnorms[:sz]
    txt_gnorms = txt_gnorms[:sz]

    print(recorded_gnorms - txt_gnorms)
    assert np.allclose(
        recorded_gnorms,
        txt_gnorms,
        rtol=1e-4,
        atol=1e-4,
    ), "GNORM comparison failed!"

    print("GNORM comparison passed!")


def test_rosenbrock():
    prob = RosenbrockProblem()
    trial(prob, 70)


def test_powell():
    prob = PowellProblem()
    trial(prob, 20)  # Whether we use daxpy or numpy, the results differ significantly.


def test_dixon_price():
    prob = DixonPriceProblem()
    prob.x0 = np.array([0.5] * prob.n, dtype=np.float64)  # todo: remove this line
    trial(prob, 100)


def test_zakharov():
    prob = ZakharovProblem()
    trial(prob, 100)


if __name__ == "__main__":
    test_rosenbrock()
    test_powell()
    test_dixon_price()
    test_zakharov()
