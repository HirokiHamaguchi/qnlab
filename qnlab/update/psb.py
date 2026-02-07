import numpy as np
import numpy.typing as npt

from qnlab.update.base import BaseUpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory


def compute_B(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    scale = lm.get_last().ys / lm.get_last().yy

    # B: Hessian approximation
    B = 1.0 / scale * np.eye(len(g))
    for item in lm:
        s = item.s
        y = item.y
        ss = np.dot(s, s)
        y_Bs = y - B @ s
        first = (np.outer(y_Bs, s) + np.outer(s, y_Bs)) / ss
        second = (np.inner(s, y_Bs) / (ss**2)) * np.outer(s, s)
        B += first - second

    return B


class PSBUpdateRule(BaseUpdateRule):
    @staticmethod
    def compute_dir(x, g, lm) -> npt.NDArray[np.float64]:
        B = compute_B(g, lm)
        return np.linalg.solve(B, -g).astype(np.float64)

    @staticmethod
    def compute_dir_reg(x, g, lm, mu) -> npt.NDArray[np.float64]:
        B = compute_B(g, lm)
        return np.linalg.solve(B + mu * np.eye(g.shape[0]), -g).astype(np.float64)

    @staticmethod
    def check(n, g, d, lm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        B = compute_B(g, lm)
        H = np.linalg.inv(B).astype(np.float64)
        d_true = H @ (-g)
        assert np.allclose(d_true, d, atol=1e-6, rtol=1e-6)
        return B, H
