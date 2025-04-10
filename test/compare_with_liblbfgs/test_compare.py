import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os

from lbfgs_lab.lbfgs import lbfgs
from lbfgs_lab.problems.rosenbrock import RosenbrockProblem


def main():
    prob = RosenbrockProblem()
    info, fx, x_opt = lbfgs(prob)
    print(info)
    return

    recorded_gnorms = np.array(prob.gnorms)
    print(f"Recorded GNORMs: {recorded_gnorms}")

    folder_path = os.path.dirname(os.path.abspath(__file__))
    file_name = "data/RosenbrockFunction_results.txt"
    with open(os.path.join(folder_path, file_name), "r") as f:
        txt_gnorms = np.array([float(line.strip()) for line in f if line.strip()])

    assert np.allclose(
        recorded_gnorms,
        txt_gnorms,
        rtol=1e-5,
        atol=1e-8,
    ), "GNORM comparison failed!"

    print("GNORM comparison passed!")


if __name__ == "__main__":
    main()
