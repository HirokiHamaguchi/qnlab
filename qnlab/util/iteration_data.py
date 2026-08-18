from typing import Tuple

import numpy as np
import numpy.typing as npt

from qnlab.util.method import Method

CAUTIOUS_CURVATURE_LOWER = np.float64(1e-8)
CAUTIOUS_CURVATURE_UPPER = np.float64(1e8)
CAUTIOUS_ABSOLUTE_FLOOR = np.float64(1e-16)


class IterationData:
    """Data structure to store per-iteration vectors and scalars."""

    def __init__(
        self,
        s: npt.NDArray[np.float64] = np.array([], dtype=np.float64),
        y: npt.NDArray[np.float64] = np.array([], dtype=np.float64),
    ):
        self.s = s  # point difference (Step)
        self.y = y  # gradient difference
        self.ss: np.float64 = np.float64(0.0)  # s^T s
        self.ys: np.float64 = np.float64(0.0)  # y^T s
        self.yy: np.float64 = np.float64(0.0)  # y^T y

        if s.size > 0 and y.size > 0:
            assert s.shape == y.shape
            self.ss = np.dot(self.s, self.s)
            self.ys = np.dot(self.y, self.s)
            self.yy = np.dot(self.y, self.y)

        # for the two-loop recursion
        self.alpha = np.float64(0.0)

    def set(
        self,
        x: npt.NDArray[np.float64],
        fx: np.float64,
        g: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fp: np.float64,
        gp: npt.NDArray[np.float64],
        method: Method,
        eps: np.float64,
    ) -> Tuple[bool, str]:
        self.s = x - xp
        self.y = g - gp
        self.ss = np.dot(self.s, self.s)
        self.yy = np.dot(self.y, self.y)
        if not np.isfinite(self.ss) or not np.isfinite(self.yy) or self.ss <= 0.0:
            return False, "infinite or invalid curvature pair"

        # Apply modified secant equation if needed
        is_reliable = (
            np.isfinite(fx)
            and np.isfinite(fp)
            and fx < fp - 2 * eps * (1 - eps) * max(1.0, np.abs(fp), np.abs(fx))
        )
        if is_reliable and method.secant in ("modified", "damped_modified"):
            sigma = self.compute_sigma(self.s, self.ss, g, gp, fx, fp)

            # ensure y^T s > 0 for line search methods
            if method.base == "Line":
                sigma = max(sigma, -0.5 * self.ys / self.ss)

            self.y = self.y + sigma * self.s
            self.yy = np.dot(self.y, self.y)
            if not np.isfinite(self.yy):
                return False, "infinite or invalid modified gradient difference"

        # Apply damped BFGS update if needed
        if method.secant in ("damped", "damped_modified"):
            # new_y = theta y + (1 - theta) Bs
            # new_y s = theta ys + (1 - theta) sBs = sBs + theta (ys - sBs) > 0
            # theta = 0.8 sBs / (sBs - ys)
            self.ys = np.dot(self.y, self.s)
            if self.ys < 0.0:
                if not np.isfinite(self.ys) or not np.isfinite(self.yy):
                    return False, "infinite or invalid damped update"
                B_scalar = abs(self.yy / self.ys)
                if B_scalar > 1e3:
                    return False, "too large damped update"
                theta_denom = self.ss * B_scalar - self.ys
                if theta_denom == 0.0:
                    return False, "infinite or invalid damped update"
                theta = (0.8 * self.ss * B_scalar) / theta_denom
                assert theta < 1.0
                self.y = theta * self.y + (1 - theta) * B_scalar * self.s
                self.ys = np.dot(self.y, self.s)
                self.yy = np.dot(self.y, self.y)
        else:
            self.ys = np.dot(self.y, self.s)

        # Check if we should store this vector based on the store rule
        if method.store == "cautious":
            curvature_floor = max(
                CAUTIOUS_CURVATURE_LOWER * self.ss,
                self.yy / CAUTIOUS_CURVATURE_UPPER,
                CAUTIOUS_ABSOLUTE_FLOOR,
            )
            if self.ys < curvature_floor:
                return False, "skip by cautious update"

        return True, ""

    def __repr__(self):
        return f"IterationData(ys={self.ys}, yy={self.yy})"

    @staticmethod
    def compute_sigma(s, ss, g, gp, fx, fp):
        # fp<fxの場合が存在する。普通にBFGSをやるなら、
        # fx側を基準としてquadratic modelが作られるが、
        # それだとfx,gxが適合して、より最適解に近いはずのfp側が不適合になってしまう。
        # つまり、fpの方が最適解に近そうなら、gpをいじるべきではない。
        # よって、ここではBFGSの方が良い
        assert fp > fx
        diff_f = fp - fx
        s_ggp = np.dot(s, g + gp)
        # Zhang et al.
        sigma_Z = (6 * diff_f + 3 * s_ggp) / ss
        # Yuan et al.
        sigma_Y = max(0.0, 2.0 * diff_f + s_ggp) / ss
        sigma = (sigma_Y + sigma_Z) / 2
        return sigma
