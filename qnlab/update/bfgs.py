import numpy as np
import numpy.typing as npt
from scipy.linalg.blas import daxpy
from scipy.linalg.lapack import dpotrs

from qnlab.update.base import BaseUpdateRule
from qnlab.util.memory_interface import QuasiNewtonMemory


def _solve_cholesky(factor, rhs):
    solution, info = dpotrs(factor, rhs, lower=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"Cholesky solve failed (LAPACK info={info}).")
    return solution


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
        shifted_gradients = workspace.shifted_gradients(mu)
        if shifted_gradients is not None:
            gradients = shifted_gradients
        uncached_shift = mu != 0.0 and shifted_gradients is None
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
            if uncached_shift:
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
            if uncached_shift:
                numerator += mu * np.dot(steps[:, index], d)
            beta = numerator / denominator
            # daxpy computes d <- d + (alpha - beta)*s without an allocation.
            d = daxpy(steps[:, index], d, a=alphas[index] - beta)
        return d

    @staticmethod
    def compute_dir_reg(x, g, lm, mu) -> npt.NDArray[np.float64]:
        """Use the shifted-pair approximation ``new_y = y + mu * s``."""
        assert len(lm) > 0
        return BFGSUpdateRule._compute_dir_reg(g, lm, mu)

    @staticmethod
    def compute_dir_additive_reg(x, g, lm, mu) -> npt.NDArray[np.float64]:
        """Solve ``(B + mu*I) d = -g`` using a compact L-BFGS formula.

        Here ``B`` is the raw L-BFGS Hessian obtained from the stored pairs.
        The normalized compact factorization follows Kanzow and Steck's
        regularized L-BFGS implementation.
        """
        del x
        assert len(lm) > 0
        if mu < 0.0:
            raise ValueError("The additive regularization must be non-negative.")
        if mu == 0.0:
            return BFGSUpdateRule.compute_dir(None, g, lm)

        workspace = lm.workspace
        indices = workspace.indices
        step_norms = workspace.step_norms
        pair_products = workspace.pair_products
        gradient_norms = workspace.gradient_norms

        step_scales = np.sqrt(step_norms)
        gradient_scales = np.sqrt(gradient_norms)
        if (
            np.any(step_scales <= 0.0)
            or np.any(gradient_scales <= 0.0)
            or np.any(pair_products <= 0.0)
        ):
            raise np.linalg.LinAlgError("Invalid L-BFGS curvature pair.")

        normalized_steps, normalized_gradients = workspace.normalized_vectors
        step_products = workspace.step_products / np.outer(step_scales, step_scales)
        step_gradient = workspace.step_gradient / np.outer(step_scales, gradient_scales)
        gradient_products = workspace.gradient_products / np.outer(
            gradient_scales, gradient_scales
        )

        gamma = gradient_norms[-1] / pair_products[-1]
        shifted_gamma = gamma + mu
        q11 = shifted_gamma * np.diag(pair_products / gradient_norms)
        q11 += gradient_products
        q21 = np.triu(step_gradient)
        q21 -= (mu / gamma) * np.tril(step_gradient, k=-1)
        q22 = -(mu / gamma) * step_products

        chol_q11 = np.linalg.cholesky(q11)
        q11_inv_q21_t = _solve_cholesky(chol_q11, q21.T)
        schur_complement = q22 - q21 @ q11_inv_q21_t
        chol_negative_schur = np.linalg.cholesky(-schur_complement)
        # Block elimination avoids constructing two 2m-by-2m triangular factors.
        gradient_solution = _solve_cholesky(
            chol_q11, (normalized_gradients.T @ g)[indices]
        )
        step_solution = _solve_cholesky(
            chol_negative_schur,
            q21 @ gradient_solution - (normalized_steps.T @ g)[indices],
        )
        gradient_solution -= q11_inv_q21_t @ step_solution
        gradient_coefficients = np.empty(len(lm), dtype=np.float64)
        step_coefficients = np.empty(len(lm), dtype=np.float64)
        gradient_coefficients[indices] = gradient_solution
        step_coefficients[indices] = step_solution
        correction = (
            normalized_gradients @ gradient_coefficients
            + normalized_steps @ step_coefficients
        )
        direction = (correction - g) / shifted_gamma
        if not np.all(np.isfinite(direction)):
            raise np.linalg.LinAlgError(
                "The compact regularized L-BFGS solve produced non-finite values."
            )
        return direction.astype(np.float64, copy=False)

    @staticmethod
    def check(n, g, d, lm) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        B, H = compute_BH(n, lm)

        assert np.allclose(np.eye(n), H @ B, atol=1e-5, rtol=1e-5), H @ B

        d_true = -H @ g
        assert np.allclose(d_true, d, atol=1e-6, rtol=1e-6)

        return B, H
