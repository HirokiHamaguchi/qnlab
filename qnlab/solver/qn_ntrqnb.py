"""Limited-memory NTRQN solver for box-constrained optimization."""

from collections import deque
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize._linesearch import _cubicmin, _quadmin  # type: ignore[import-untyped]

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.util.callback import Callback
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode

Array = npt.NDArray[np.float64]


def _prepare_bounds(
    bounds: Union[Sequence[tuple[float, float]], object], n: int
) -> tuple[Array, Array]:
    """Convert scipy-style bounds to finite-or-infinite float arrays."""
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lb = np.broadcast_to(np.asarray(bounds.lb, dtype=np.float64), (n,)).copy()
        ub = np.broadcast_to(np.asarray(bounds.ub, dtype=np.float64), (n,)).copy()
    else:
        if len(bounds) != n:  # type: ignore[arg-type]
            raise ValueError(f"bounds must contain {n} (lower, upper) pairs")
        lb = np.array(
            [-np.inf if pair[0] is None else pair[0] for pair in bounds],  # type: ignore[union-attr]
            dtype=np.float64,
        )
        ub = np.array(
            [np.inf if pair[1] is None else pair[1] for pair in bounds],  # type: ignore[union-attr]
            dtype=np.float64,
        )
    if np.any(np.isnan(lb)) or np.any(np.isnan(ub)) or np.any(lb > ub):
        raise ValueError("each lower bound must be less than or equal to its upper bound")
    return lb, ub


def projected_gradient(x: Array, g: Array, lb: Array, ub: Array) -> Array:
    """Return ``x - P_[lb, ub](x - g)``."""
    return x - np.clip(x - g, lb, ub)


def max_feasible_step(x: Array, d: Array, lb: Array, ub: Array) -> np.float64:
    """Largest nonnegative step for which ``x + alpha*d`` stays feasible."""
    alpha_max = np.float64(np.inf)
    pos = d > 0.0
    if np.any(pos):
        alpha_max = np.minimum(alpha_max, np.min((ub[pos] - x[pos]) / d[pos]))
    neg = d < 0.0
    if np.any(neg):
        alpha_max = np.minimum(alpha_max, np.min((lb[neg] - x[neg]) / d[neg]))
    return np.float64(np.maximum(0.0, alpha_max))


@dataclass
class _CompactBFGS:
    """Compact representation of ``B + mu I`` from full-space pairs."""

    diagonal: np.float64
    vectors: Array
    coefficients: Array

    @classmethod
    def from_memory(cls, lm: QuasiNewtonMemory, mu: np.float64, n: int):
        threshold = np.finfo(float).eps
        items = [item for item in lm if item.ys > threshold * item.ss]
        theta = (
            np.float64(items[-1].yy / items[-1].ys)
            if items
            else np.float64(1.0 / lm.zero_length)
        )
        theta = np.float64(np.maximum(theta, np.finfo(float).tiny))

        vectors: list[Array] = []
        coefficients: list[np.float64] = []
        for item in items:
            bs = theta * item.s.copy()
            for vector, coefficient in zip(vectors, coefficients):
                bs += coefficient * vector * np.dot(vector, item.s)
            sbs = np.dot(item.s, bs)
            if not np.isfinite(sbs) or sbs <= threshold * item.ss:
                continue
            vectors.extend((bs, item.y))
            coefficients.extend((np.float64(-1.0 / sbs), np.float64(1.0 / item.ys)))

        matrix = np.column_stack(vectors) if vectors else np.empty((n, 0))
        return cls(theta + mu, matrix, np.asarray(coefficients, dtype=np.float64))

    def apply(self, v: Array) -> Array:
        result = self.diagonal * v
        if self.vectors.shape[1]:
            result = result + self.vectors @ (
                self.coefficients * (self.vectors.T @ v)
            )
        return result

    def solve_restricted(self, rhs: Array, free: npt.NDArray[np.bool_]) -> Array:
        """Solve the principal free-variable system using Woodbury."""
        result = np.zeros_like(rhs)
        if not np.any(free):
            return result
        restricted_rhs = rhs[free]
        restricted_vectors = self.vectors[free]
        solution = restricted_rhs / self.diagonal
        if restricted_vectors.shape[1]:
            middle = np.diag(1.0 / self.coefficients)
            middle += restricted_vectors.T @ restricted_vectors / self.diagonal
            right = restricted_vectors.T @ restricted_rhs / self.diagonal
            try:
                correction = np.linalg.solve(middle, right)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(middle, right, rcond=None)[0]
            solution -= restricted_vectors @ correction / self.diagonal
        result[free] = solution
        return result


def _generalized_cauchy_point(
    x: Array, g: Array, lb: Array, ub: Array, operator: _CompactBFGS
) -> tuple[Array, npt.NDArray[np.bool_]]:
    """Minimize the quadratic model along the projected-gradient path."""
    p = np.zeros_like(x)
    d = -g.copy()
    active = (lb == ub) | ((x <= lb) & (g > 0.0)) | ((x >= ub) & (g < 0.0))
    d[active] = 0.0

    breakpoints: list[tuple[float, int]] = []
    for i in np.flatnonzero(d):
        distance = ub[i] - x[i] if d[i] > 0.0 else lb[i] - x[i]
        step = distance / d[i]
        if np.isfinite(step) and step >= 0.0:
            breakpoints.append((float(step), int(i)))
    breakpoints.sort()

    hit = active.copy()
    previous = 0.0
    cursor = 0
    while cursor < len(breakpoints):
        breakpoint = breakpoints[cursor][0]
        interval = max(0.0, breakpoint - previous)
        model_gradient = g + operator.apply(p)
        derivative = np.dot(model_gradient, d)
        curvature = np.dot(d, operator.apply(d))
        minimizer = -derivative / curvature if curvature > 0.0 else np.inf
        if 0.0 <= minimizer < interval:
            p += minimizer * d
            return np.clip(x + p, lb, ub), hit
        p += interval * d

        while cursor < len(breakpoints) and breakpoints[cursor][0] == breakpoint:
            i = breakpoints[cursor][1]
            p[i] = (ub[i] if d[i] > 0.0 else lb[i]) - x[i]
            d[i] = 0.0
            hit[i] = True
            cursor += 1
        previous = breakpoint

    if np.any(d):
        model_gradient = g + operator.apply(p)
        curvature = np.dot(d, operator.apply(d))
        if curvature > 0.0:
            p += max(0.0, -np.dot(model_gradient, d) / curvature) * d
    return np.clip(x + p, lb, ub), hit


def _search_direction(
    x: Array, g: Array, lb: Array, ub: Array, operator: _CompactBFGS
) -> Array:
    cauchy, active = _generalized_cauchy_point(x, g, lb, ub, operator)
    p = cauchy - x
    free = ~active
    reduced_gradient = g + operator.apply(p)
    correction = operator.solve_restricted(-reduced_gradient, free)
    candidate = np.clip(cauchy + correction, lb, ub)
    direction = candidate - x

    # L-BFGS-B likewise safeguards a projected subspace step by descent.
    if np.dot(g, direction) >= 0.0:
        direction = p
    if np.dot(g, direction) >= 0.0:
        direction = -projected_gradient(x, g, lb, ub)
        direction[(x <= lb) & (direction < 0.0)] = 0.0
        direction[(x >= ub) & (direction > 0.0)] = 0.0
    return direction


def _line_search(
    x: Array,
    fx: np.float64,
    g: Array,
    d: Array,
    lb: Array,
    ub: Array,
    prob: BaseProblem,
    param: NTRQNParameter,
    eps: np.float64,
    ref_fx: np.float64,
    is_offo_mode: bool,
    rejection_counter: int,
) -> tuple[RetCode, Array, np.float64, Array, np.float64, int]:
    alpha = np.float64(min(1.0, max_feasible_step(x, d, lb, ub)))
    dg = np.dot(d, g)
    if not dg < 0.0 or alpha <= 0.0:
        return RetCode.ERR_INCREASEGRADIENT, x, fx, g, np.float64(np.inf), rejection_counter

    previous_alpha = None
    previous_f = None
    for _ in range(param.max_linesearch):
        x_try = x + alpha * d
        f_try = prob.f(x_try)
        delta = 2 * eps / (1 - eps) * max(fx, -f_try, 1.0)
        if not np.isfinite(f_try) or ref_fx + param.armijo * alpha * dg + delta < f_try:
            old_alpha = alpha
            best_alpha = None
            if np.isfinite(f_try) and previous_alpha is not None:
                candidate = _cubicmin(
                    np.float64(0.0), fx, dg, old_alpha, f_try,
                    previous_alpha, previous_f,
                )
                if candidate is not None and 0.2 * old_alpha < candidate < 0.8 * old_alpha:
                    best_alpha = candidate
            if np.isfinite(f_try) and best_alpha is None:
                candidate = _quadmin(np.float64(0.0), fx, dg, old_alpha, f_try)
                if candidate is not None and 0.1 * old_alpha < candidate < 0.9 * old_alpha:
                    best_alpha = candidate
            alpha = np.float64(best_alpha if best_alpha is not None else 0.5 * old_alpha)
            previous_alpha, previous_f = old_alpha, f_try
            continue

        g_try = prob.g(x_try)
        if not np.all(np.isfinite(g_try)):
            alpha /= 4.0
            rejection_counter += 1
            if rejection_counter >= param.max_inf_nan_rejections:
                return RetCode.ERR_NUMERICAL_INSTABILITY, x_try, f_try, g_try, delta, rejection_counter
            continue

        pg_try_norm = np.linalg.norm(projected_gradient(x_try, g_try, lb, ub))
        if is_offo_mode and alpha == 1.0 and pg_try_norm > param.gtol:
            dg_try = np.dot(d, g_try)
            if dg_try > 0.5 * np.linalg.norm(d) * pg_try_norm:
                alpha *= np.clip(1 / 16, -dg / (dg_try - dg), 15 / 16)
                continue
        return RetCode.SUCCESS, x_try, f_try, g_try, delta, rejection_counter

    return RetCode.ERR_MAXIMUMLINESEARCH, x, fx, g, np.float64(np.inf), rejection_counter


def qn_ntrqnb(
    prob: BaseProblem,
    bounds: Union[Sequence[tuple[float, float]], object],
    param: NTRQNParameter,
    method: Method,
    callback: Union[Callback, None] = None,
    verbose: bool = False,
) -> tuple[RetCode, np.float64, Array]:
    """Minimize a differentiable objective subject to box constraints."""
    if method.update != "bfgs":
        raise ValueError("NTRQNB currently supports only the BFGS update")
    prob.reset()
    eps = prob.get_machine_eps()
    lb, ub = _prepare_bounds(bounds, prob.n)
    x = np.clip(np.asarray(prob.x0, dtype=np.float64), lb, ub)
    if callback:
        callback.start(prob, x)

    fx = prob.f(x)
    g = prob.g(x)
    pg = projected_gradient(x, g, lb, ub)
    pg_norm = np.float64(np.linalg.norm(pg))
    if not np.isfinite(pg_norm) or not np.isfinite(fx):
        return RetCode.ERR_NUMERICAL_OVERFLOW, fx, x
    if np.linalg.norm(pg, ord=np.inf) <= param.gtol:
        return RetCode.ALREADY_MINIMIZED, fx, x

    memory = QuasiNewtonMemory(pg, param.m, method)
    past_fx: deque[np.float64] = deque([], maxlen=param.past)
    reference_values: deque[np.float64] = deque([fx], maxlen=param.non_monotone)
    mu = np.float64(0.0)
    offo = np.float64(1e-10)
    min_fx_minus_delta = np.float64(np.inf)
    is_offo_mode = False
    rejection_counter = 0
    k = 0

    while True:
        if min_fx_minus_delta >= fx:
            is_offo_mode = False
            mu = np.float64(0.0)
            if min_fx_minus_delta - fx >= 1.0:
                offo = np.float64(1e-10)
        else:
            is_offo_mode = True
            offo = np.sqrt(offo**2 + pg_norm**2)
            mu = np.clip(
                pg_norm * param.mu_scale,
                param.mu_min_fraction * offo,
                offo,
            )

        operator = _CompactBFGS.from_memory(memory, mu, prob.n)
        direction = _search_direction(x, g, lb, ub, operator)
        result = _line_search(
            x, fx, g, direction, lb, ub, prob, param, eps,
            np.max(reference_values), is_offo_mode, rejection_counter,
        )
        code, new_x, new_fx, new_g, delta, rejection_counter = result
        if code != RetCode.SUCCESS:
            if callback:
                callback.callback(prob, x, fx, g)
            return code, fx, x

        memory.add_new_data(new_x, new_fx, new_g, x, fx, g, callback, eps)
        if mu == 0.0:
            min_fx_minus_delta = np.minimum(min_fx_minus_delta, fx - delta)
        x, fx, g = new_x, new_fx, new_g
        k += 1
        pg = projected_gradient(x, g, lb, ub)
        pg_norm = np.float64(np.linalg.norm(pg))

        if callback:
            callback.callback(prob, x, fx, g)
        if verbose:
            print(f"iter:{k:04} mu:{mu:.2e} f:{fx:.6e} |pg|:{pg_norm:.2e}")

        if 0 < param.past <= len(past_fx):
            if abs(np.max(past_fx) - fx) < param.ftol * abs(fx):
                return RetCode.STOP, fx, x
        past_fx.append(fx)
        if np.linalg.norm(pg, ord=np.inf) <= param.gtol:
            return RetCode.SUCCESS, fx, x
        if not np.isfinite(pg_norm) or not np.isfinite(fx):
            return RetCode.ERR_NUMERICAL_OVERFLOW, fx, x
        if prob.call_f >= param.max_evaluations:
            return RetCode.ERR_MAXIMUMEVALUATION, fx, x
        if 0 < param.max_iterations <= k:
            return RetCode.ERR_MAXIMUMITERATION, fx, x
        reference_values.append(fx)
