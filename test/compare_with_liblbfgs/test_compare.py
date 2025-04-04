from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.optimize

from lbfgs_lab.LBFGS._objectiveFunction import ObjectiveFunction
from lbfgs_lab.LBFGS.lbfgs import lbfgs


def solve_by_liblbfgs() -> npt.NDArray[np.float64]:
    # By running liblbfgs/sample/sample, we get the following data:
    x_values = np.array(
        [
            [-1.069065, 1.053443],
            [-1.037018, 1.067078],
            [-1.030061, 1.068398],
            [-1.028564, 1.066942],
            [-1.014182, 1.047120],
            [-0.982345, 0.995084],
            [-0.581753, 0.301077],
            [-0.597952, 0.329674],
            [-0.541852, 0.269778],
            [-0.314198, 0.034623],
            [-0.332048, 0.112318],
            [-0.211875, 0.031346],
            [-0.137272, -0.010314],
            [-0.049321, -0.033664],
            [0.044264, -0.018526],
            [0.157071, 0.002165],
            [0.282792, 0.093053],
            [0.381547, 0.132559],
            [0.464638, 0.185368],
            [0.464381, 0.202228],
            [0.537234, 0.283310],
            [0.590560, 0.338574],
            [0.655819, 0.415627],
            [0.724676, 0.514231],
            [0.770742, 0.595941],
            [0.824339, 0.674536],
            [0.893123, 0.784686],
            [0.881711, 0.773804],
            [0.909484, 0.826115],
            [0.966817, 0.930243],
            [0.967873, 0.935752],
            [0.988365, 0.976425],
            [0.997686, 0.995114],
            [0.999517, 0.999070],
            [1.000091, 1.000175],
            [0.999999, 0.999999],
            [1.000000, 1.000000],
        ]
    )
    return x_values


def solve_by_scipy(x0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    obj = ObjectiveFunction()
    hist: List[npt.NDArray[np.float64]] = []

    def callback(xk: npt.NDArray[np.float64]) -> bool:
        hist.append(xk)
        return False

    scipy.optimize.minimize(
        obj.evaluate,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
    )

    return np.array(hist)


def main():
    n = 100

    x0 = np.empty(n)
    for i in range(0, n, 2):
        x0[i] = -1.2
        x0[i + 1] = 1.0

    # x_liblbfgs = solve_by_liblbfgs()
    # x_scipy = solve_by_scipy(x0)
    # fig, ax = plt.subplots()
    # ax.plot(y_liblbfgs, label="liblbfgs")
    # ax.plot(y_scipy, label="scipy")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_title("L-BFGS Comparison")
    # ax.legend()
    # # plt.show()
    # plt.savefig("lbfgs.png")

    obj = ObjectiveFunction()
    info, fx, x_opt = lbfgs(n, x0, obj, None)

    print(f"{info=}")
    print(f"{fx=}")
    print(f"{x_opt=}")


if __name__ == "__main__":
    main()
