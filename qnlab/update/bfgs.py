import numpy as np
import numpy.typing as npt
from scipy.linalg.blas import daxpy

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
    def compute_dir(x, g, lm) -> npt.NDArray[np.float64]:
        # Recursive formula to compute dir = -(H \cdot g).
        # This is described in page 779 of:
        # Jorge Nocedal.
        # Updating Quasi-Newton Matrices with Limited Storage.
        # Mathematics of Computation, Vol. 35, No. 151,
        # pp. 773--782, 1980.
        if len(lm) == 0:
            return -g
        return BFGSUpdateRule._compute_dir_reg(g, lm, np.float64(0.0))

    @staticmethod
    def _compute_dir_reg(g, lm, mu) -> npt.NDArray[np.float64]:
        workspace = lm.workspace
        steps = workspace._steps
        gradients = workspace._gradients
        step_norms = workspace._step_norms
        pair_products = workspace._pair_products
        gradient_norms = workspace._gradient_norms
        alphas = workspace.alphas

        d = -g.copy()
        # Follow the ring buffer's logical order without rearranging its columns.
        indices = workspace.indices
        for index in indices[::-1]:
            denominator = pair_products[index] + mu * step_norms[index]
            alpha = np.dot(steps[:, index], d) / denominator
            alphas[index] = alpha
            # daxpy computes d <- d - alpha*y in place, avoiding an n-vector temporary.
            d = daxpy(gradients[:, index], d, a=-alpha)
            if mu != 0.0:
                d = daxpy(steps[:, index], d, a=-alpha * mu)

        last = workspace.last_index
        numerator = pair_products[last] + mu * step_norms[last]
        denominator = (
            gradient_norms[last]
            + 2.0 * mu * pair_products[last]
            + mu * mu * step_norms[last]
        )
        d *= numerator / denominator

        for index in indices:
            denominator = pair_products[index] + mu * step_norms[index]
            numerator = np.dot(gradients[:, index], d)
            if mu != 0.0:
                numerator += mu * np.dot(steps[:, index], d)
            beta = numerator / denominator
            # daxpy computes d <- d + (alpha - beta)*s without an allocation.
            d = daxpy(steps[:, index], d, a=alphas[index] - beta)
        return d

    @staticmethod
    def compute_dir_reg(x, g, lm, mu) -> npt.NDArray[np.float64]:
        """Uses new_y = y + mu * s in the update."""
        assert len(lm) > 0
        return BFGSUpdateRule._compute_dir_reg(g, lm, mu)

    @staticmethod
    def check(n, g, d, lm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        B, H = compute_BH(n, lm)

        assert np.allclose(np.eye(n), H @ B, atol=1e-5, rtol=1e-5), H @ B

        d_true = -H @ g
        assert np.allclose(d_true, d, atol=1e-6, rtol=1e-6)

        return B, H
