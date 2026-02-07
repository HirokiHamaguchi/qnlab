import numpy as np
import numpy.typing as npt

from qnlab.update.base import BaseUpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory


def compute_B(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    scale = lm.get_last().ys / lm.get_last().yy
    B = 1.0 / scale * np.eye(len(g))
    for item in lm:
        s = item.s
        y = item.y
        Bs = B @ s
        rho = 1.0 / item.ys
        B += rho * (
            -(np.outer(y, Bs) + np.outer(Bs, y))
            + (np.dot(s, Bs) * rho + 1.0) * np.outer(y, y)
        )
    return B


def compute_H(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    scale = lm.get_last().ys / lm.get_last().yy
    H = scale * np.eye(len(g))
    for item in lm:
        s = item.s
        y = item.y
        Hy = H @ y
        H -= np.outer(Hy, Hy) / np.dot(y, Hy)
        H += np.outer(s, s) / item.ys
    return H


class DFPUpdateRule(BaseUpdateRule):
    @staticmethod
    def compute_dir(x, g, lm) -> npt.NDArray[np.float64]:
        H = compute_H(g, lm)
        return -H @ g

    @staticmethod
    def check(n, g, d, lm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        B = compute_B(g, lm)
        H = compute_H(g, lm)
        assert np.allclose(np.eye(n), H @ B, atol=1e-5, rtol=1e-5), H @ B

        d_true = -H @ g
        assert np.allclose(d_true, d, atol=1e-6, rtol=1e-6)

        return B, H
