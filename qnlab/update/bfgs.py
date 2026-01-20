import numpy as np
import numpy.typing as npt

from qnlab.update.base import BaseUpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory


def compute_BH(
    n: int,
    lm: QuasiNewtonMemory,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    scale = lm.get_last().ys / lm.get_last().yy

    # B: Hessian approximation
    B = 1.0 / scale * np.eye(n)
    for item in lm:
        s = item.s
        y = item.y
        Bs = B @ s
        B -= np.outer(Bs, Bs) / np.dot(s, Bs)
        B += np.outer(y, y) / item.ys

    # H: inverse Hessian approximation
    H = scale * np.eye(n)
    for item in lm:
        s = item.s
        y = item.y
        Hy = H @ y
        rho = 1.0 / item.ys
        H += rho * (
            -(np.outer(Hy, s) + np.outer(s, Hy))
            + (np.dot(y, Hy) * rho + 1.0) * np.outer(s, s)
        )

    return B, H


class BFGSUpdateRule(BaseUpdateRule):
    @staticmethod
    def compute_dir(x, fx, g, lm) -> npt.NDArray[np.float64]:
        # Recursive formula to compute dir = -(H \cdot g).
        # This is described in page 779 of:
        # Jorge Nocedal.
        # Updating Quasi-Newton Matrices with Limited Storage.
        # Mathematics of Computation, Vol. 35, No. 151,
        # pp. 773--782, 1980.
        if len(lm) == 0:
            return -g
        d = -g.copy()
        for item in reversed(lm):
            item.alpha = np.dot(item.s, d) / item.ys
            d -= item.alpha * item.y
        d *= lm.get_last().ys / lm.get_last().yy
        for item in lm:
            beta = np.dot(item.y, d) / item.ys
            d += (item.alpha - beta) * item.s
        return d

    @staticmethod
    def compute_dir_reg(x, fx, g, lm, mu) -> npt.NDArray[np.float64]:
        """Uses new_y = y + mu * s in the update."""
        assert len(lm) > 0
        d = -g.copy()
        new_y_ys = []
        for item in reversed(lm):
            new_y = item.y + mu * item.s
            new_y_ys.append(new_y)
            item.alpha = np.dot(item.s, d) / (item.ys + mu * item.ss)
            d -= item.alpha * new_y
        firstItem = lm.get_last()
        d *= (firstItem.ys + mu * firstItem.ss) / (
            firstItem.yy + 2.0 * mu * firstItem.ys + mu * mu * firstItem.ss
        )
        for item, new_y in zip(lm, reversed(new_y_ys)):
            beta = np.dot(new_y, d) / (item.ys + mu * item.ss)
            d += (item.alpha - beta) * item.s
        return d

    @staticmethod
    def check(n, g, d, lm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        B, H = compute_BH(n, lm)

        assert np.allclose(np.eye(n), H @ B, atol=1e-5, rtol=1e-5), H @ B

        d_true = -H @ g
        assert np.allclose(d_true, d, atol=1e-6, rtol=1e-6)

        return B, H
