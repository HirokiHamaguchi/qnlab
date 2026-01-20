import numpy as np
import numpy.typing as npt

from qnlab.update.update import check_direction, get_direction, get_direction_reg
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method


def generate_lm(
    n, maxlen=3
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    QuasiNewtonMemory,
]:
    lm = QuasiNewtonMemory(np.zeros(n), maxlen=3, method=Method())

    A = np.random.randn(n, n)
    A = A @ A.T  # SPD
    b = np.random.randn(n)
    c = np.random.randn()

    def f(x) -> np.float64:
        return 0.5 * x @ A @ x + b @ x + c

    xp = np.random.randn(n)
    gp = np.random.randn(n)

    while True:
        # generate a nonzero s vector
        xk = np.random.randn(n)
        gk = np.random.randn(n)

        lm.add_new_data(xk, f(xk), gk, xp, f(xp), gp, None, np.float64(0.0))

        if len(lm) and (
            np.abs(lm.get_last().ys) < 1e-3 or np.abs(lm.get_last().yy) < 1e-3
        ):
            lm._deque.pop()
            continue

        if len(lm) >= maxlen:
            return xk, gk, lm


def test_dir_BFGS_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "bfgs")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_lbfgs = get_direction(method, xk, np.float64(0.0), gk, lm)
        check_direction(method, n, gk, d_lbfgs, lm)
    print("BFGS passed")


def test_dir_DFP_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "dfp")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_dfp = get_direction(method, xk, np.float64(0.0), gk, lm)
        check_direction(method, n, gk, d_dfp, lm)
    print("DFP passed")


def test_dir_SR1_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "sr1")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_sr1 = get_direction_reg(
            method, xk, np.float64(0.0), gk, lm, mu=np.float64(0.0)
        )
        check_direction(method, n, gk, d_sr1, lm)
    print("SR1 passed")


def test_dir_PSB_and_check():
    n = 5
    method = Method("Line", "raw", "raw", "psb")
    for _ in range(10):
        xk, gk, lm = generate_lm(n)
        d_psb = get_direction_reg(
            method, xk, np.float64(0.0), gk, lm, mu=np.float64(0.0)
        )
        check_direction(method, n, gk, d_psb, lm)
    print("PSB passed")


if __name__ == "__main__":
    np.random.seed(0)  # Since numerically instable, we set a seed for reproducibility
    test_dir_BFGS_and_check()
    test_dir_DFP_and_check()
    test_dir_SR1_and_check()
    test_dir_PSB_and_check()
