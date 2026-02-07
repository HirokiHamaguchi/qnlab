import numpy as np
import numpy.typing as npt

from qnlab.update.base import BaseUpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory

# https://en.wikipedia.org/wiki/Symmetric_rank-one


def compute_B(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    n = g.shape[0]
    scale = lm.get_last().ys / lm.get_last().yy

    B = 1.0 / scale * np.eye(n)
    for item in lm:
        Bs = B @ item.s
        y_m_Bs = item.y - Bs
        denom = np.dot(y_m_Bs, item.s)

        # if np.abs(denom) < 1e-10:
        #     continue
        assert np.abs(denom) >= 1e-10, "SR1 check failed: denom is too small."

        B += np.outer(y_m_Bs, y_m_Bs) / denom

    return B


def compute_H(
    g: npt.NDArray[np.float64], lm: QuasiNewtonMemory
) -> npt.NDArray[np.float64]:
    n = g.shape[0]
    scale = lm.get_last().ys / lm.get_last().yy

    # H: inverse Hessian approximation
    H = scale * np.eye(n)
    for item in lm:
        s_m_Hy = item.s - H @ item.y
        denom = np.dot(s_m_Hy, item.y)

        if np.abs(denom) < 1e-10:
            # when H=\gamma I and d=-g, this condition is satisfied.
            # assert np.abs(denom) >= 1e-10, "SR1 check failed: denom is too small."
            continue

        H += np.outer(s_m_Hy, s_m_Hy) / denom

    return H


class SR1UpdateRule(BaseUpdateRule):
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
        H = compute_H(g, lm)

        assert np.allclose(np.eye(n), H @ B, atol=1e-6, rtol=1e-6)

        d_true = -H @ g
        assert np.allclose(d, d_true, atol=1e-6, rtol=1e-6)

        return B, H
