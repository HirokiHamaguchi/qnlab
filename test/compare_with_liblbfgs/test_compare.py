import numpy as np
import os

from typing import Union
from lbfgs_lab.lbfgs import lbfgs

from lbfgs_lab.problems.rosenbrock import RosenbrockProblem
from lbfgs_lab.problems.dixon_price import DixonPriceProblem
from lbfgs_lab.problems.powell import PowellProblem
from lbfgs_lab.problems.zakharov import ZakharovProblem


def trial(
    prob: Union[RosenbrockProblem, PowellProblem, DixonPriceProblem, ZakharovProblem],
):
    name = prob.__class__.__name__.replace("Problem", "")
    print(f"----- Problem name: {name} -----")

    info, fx, x_opt = lbfgs(prob)
    print(f"Info: {info}")

    recorded_gnorms = np.array(prob.gnorms)

    folder_path = os.path.dirname(os.path.abspath(__file__))
    file_name = f"data/{name}_results.txt"
    with open(os.path.join(folder_path, file_name), "r") as f:
        txt_gnorms = np.array([float(line.strip()) for line in f if line.strip()])

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
    trial(prob)


def test_zakharov():
    prob = ZakharovProblem()
    trial(prob)


if __name__ == "__main__":
    test_rosenbrock()
    test_powell()
    test_dixon_price()
    test_zakharov()
