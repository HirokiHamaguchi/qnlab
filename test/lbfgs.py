import numpy as np
from lbfgs_lab.LBFGS.lbfgs import lbfgs
from lbfgs_lab.LBFGS._objectiveFunction import ObjectiveFunction


def main():
    n = 100

    x0 = np.empty(n)
    for i in range(0, n, 2):
        x0[i] = -1.2
        x0[i + 1] = 1.0

    obj = ObjectiveFunction()
    x_opt, fx, info = lbfgs(n, x0, obj, None)

    print(f"{x_opt=}")
    print(f"{fx=}")
    print(f"{info=}")


if __name__ == "__main__":
    main()
