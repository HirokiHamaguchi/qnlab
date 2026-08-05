"""Limited-memory NTRQN solver for box-constrained optimization."""

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeAlias, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn_ntrqn import line_search_relaxed_armijo
from qnlab.update.update import get_direction_reg
from qnlab.util.callback import Callback
from qnlab.util.check_termination import check_termination
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode

BoundPair: TypeAlias = tuple[float | None, float | None]


@runtime_checkable
class BoundsLike(Protocol):
    """Structural type for objects such as ``scipy.optimize.Bounds``."""

    @property
    def lb(self) -> object: ...

    @property
    def ub(self) -> object: ...


BoundsInput: TypeAlias = Sequence[BoundPair] | BoundsLike


def prepare_bounds(
    bounds: BoundsInput, n: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert scipy-style bounds to finite-or-infinite float arrays."""
    if isinstance(bounds, BoundsLike):
        lb = np.broadcast_to(np.asarray(bounds.lb, dtype=np.float64), (n,)).copy()
        ub = np.broadcast_to(np.asarray(bounds.ub, dtype=np.float64), (n,)).copy()
    else:
        if len(bounds) != n:
            raise ValueError(f"bounds must contain {n} (lower, upper) pairs")
        lb = np.array(
            [-np.inf if pair[0] is None else pair[0] for pair in bounds],
            dtype=np.float64,
        )
        ub = np.array(
            [np.inf if pair[1] is None else pair[1] for pair in bounds],
            dtype=np.float64,
        )
    if np.any(np.isnan(lb)) or np.any(np.isnan(ub)) or np.any(lb > ub):
        raise ValueError(
            "each lower bound must be less than or equal to its upper bound"
        )
    return lb, ub


def projected_gradient(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the projected-gradient mapping ``x - P_[lb, ub](x - g)``."""
    return np.clip(g, x - ub, x - lb)


def max_feasible_step(
    x: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
) -> np.float64:
    """Largest nonnegative step for which ``x + alpha*d`` stays feasible."""
    alpha_max = np.float64(np.inf)
    positive = d > 0.0
    if np.any(positive):
        alpha_max = np.minimum(
            alpha_max, np.min((ub[positive] - x[positive]) / d[positive])
        )
    negative = d < 0.0
    if np.any(negative):
        alpha_max = np.minimum(
            alpha_max, np.min((lb[negative] - x[negative]) / d[negative])
        )
    return np.float64(np.maximum(0.0, alpha_max))


@dataclass
class _CompactBFGS:
    """Compact Hessian built from the regularized pairs ``(s, y + mu*s)``."""

    diagonal: np.float64
    vectors: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]

    @classmethod
    def from_memory(
        cls,
        memory: QuasiNewtonMemory,
        mu: np.float64,
        n: int,
    ) -> "_CompactBFGS":
        """Build the same regularized L-BFGS matrix used by NTRQN's two loops."""
        if len(memory) == 0:
            diagonal = mu if mu > 0.0 else np.float64(1.0 / memory.zero_length)
            return cls(
                np.float64(diagonal),
                np.empty((n, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        last = memory.get_last()
        last_y = last.y + mu * last.s
        last_ys = last.ys + mu * last.ss
        diagonal = np.float64(np.dot(last_y, last_y) / last_ys)

        vectors: list[npt.NDArray[np.float64]] = []
        coefficients: list[np.float64] = []
        for item in memory:
            regularized_y = item.y + mu * item.s
            regularized_ys = item.ys + mu * item.ss

            bs = diagonal * item.s
            for vector, coefficient in zip(vectors, coefficients):
                bs = bs + coefficient * vector * np.dot(vector, item.s)
            sbs = np.dot(item.s, bs)

            # Stored BFGS pairs should make both denominators positive. Skip a
            # pair if roundoff nevertheless makes its compact update unusable.
            if (
                not np.isfinite(regularized_ys)
                or not np.isfinite(sbs)
                or regularized_ys <= 0.0
                or sbs <= 0.0
            ):
                continue
            vectors.extend((bs, regularized_y))
            coefficients.extend(
                (np.float64(-1.0 / sbs), np.float64(1.0 / regularized_ys))
            )

        compact_vectors = (
            np.column_stack(vectors) if vectors else np.empty((n, 0), dtype=np.float64)
        )
        return cls(
            diagonal,
            compact_vectors,
            np.asarray(coefficients, dtype=np.float64),
        )

    def apply(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the compact Hessian approximation."""
        result = self.diagonal * vector
        if self.vectors.shape[1] > 0:
            result = result + self.vectors @ (
                self.coefficients * (self.vectors.T @ vector)
            )
        return result

    def solve_restricted(
        self,
        rhs: npt.NDArray[np.float64],
        free: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """Solve a principal free-variable system using Woodbury's identity."""
        result = np.zeros_like(rhs)
        if not np.any(free):
            return result

        restricted_rhs = rhs[free]
        restricted_vectors = self.vectors[free]
        solution = restricted_rhs / self.diagonal
        if restricted_vectors.shape[1] > 0:
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
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    operator: _CompactBFGS,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Minimize the quadratic model along the projected-gradient path."""
    step = np.zeros_like(x)
    path_direction = -g.copy()
    active = (lb == ub) | ((x <= lb) & (g > 0.0)) | ((x >= ub) & (g < 0.0))
    path_direction[active] = 0.0

    breakpoints: list[tuple[float, int]] = []
    for i in np.flatnonzero(path_direction):
        bound = ub[i] if path_direction[i] > 0.0 else lb[i]
        breakpoint = (bound - x[i]) / path_direction[i]
        if np.isfinite(breakpoint) and breakpoint >= 0.0:
            breakpoints.append((float(breakpoint), int(i)))
    breakpoints.sort()

    previous = 0.0
    cursor = 0
    while cursor < len(breakpoints):
        breakpoint = breakpoints[cursor][0]
        interval = np.float64(max(0.0, breakpoint - previous))
        model_gradient = g + operator.apply(step)
        derivative = np.dot(model_gradient, path_direction)
        curvature = np.dot(path_direction, operator.apply(path_direction))
        minimizer = (
            np.float64(-derivative / curvature)
            if curvature > 0.0
            else np.float64(np.inf)
        )
        if 0.0 <= minimizer < interval:
            step += minimizer * path_direction
            return np.clip(x + step, lb, ub), active

        step += interval * path_direction
        while cursor < len(breakpoints) and breakpoints[cursor][0] == breakpoint:
            i = breakpoints[cursor][1]
            step[i] = (ub[i] if path_direction[i] > 0.0 else lb[i]) - x[i]
            path_direction[i] = 0.0
            active[i] = True
            cursor += 1
        previous = breakpoint

    if np.any(path_direction):
        model_gradient = g + operator.apply(step)
        curvature = np.dot(path_direction, operator.apply(path_direction))
        if curvature > 0.0:
            minimizer = max(0.0, -np.dot(model_gradient, path_direction) / curvature)
            step += minimizer * path_direction
    return np.clip(x + step, lb, ub), active


def _subspace_minimization(
    cauchy: npt.NDArray[np.float64],
    active: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    operator: _CompactBFGS,
) -> npt.NDArray[np.float64]:
    """Minimize over the Cauchy point's free variables, adding hit bounds."""
    point = cauchy.copy()
    working_active = active.copy()

    # Each truncated correction fixes at least one additional variable.
    for _ in range(x.size + 1):
        free = ~working_active
        if not np.any(free):
            break
        step = point - x
        model_gradient = g + operator.apply(step)
        correction = operator.solve_restricted(-model_gradient, free)
        if not np.all(np.isfinite(correction)) or not np.any(correction[free]):
            break

        alpha = min(1.0, float(max_feasible_step(point, correction, lb, ub)))
        if alpha <= 0.0:
            break
        point += alpha * correction
        point = np.clip(point, lb, ub)
        if alpha >= 1.0 - 10.0 * np.finfo(np.float64).eps:
            break

        newly_active = free & (
            (
                (correction < 0.0)
                & np.isfinite(lb)
                & np.isclose(point, lb, rtol=1e-12, atol=1e-14)
            )
            | (
                (correction > 0.0)
                & np.isfinite(ub)
                & np.isclose(point, ub, rtol=1e-12, atol=1e-14)
            )
        )
        if not np.any(newly_active):
            break
        working_active |= newly_active
    return point


def _box_quasi_newton_direction(
    method: Method,
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    memory: QuasiNewtonMemory,
    mu: np.float64,
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Approximately minimize the regularized L-BFGS model over the box."""
    unconstrained = get_direction_reg(method, x, g, memory, mu)
    if (
        np.all(np.isfinite(unconstrained))
        and np.all(lb - x <= unconstrained)
        and np.all(unconstrained <= ub - x)
    ):
        return unconstrained

    operator = _CompactBFGS.from_memory(memory, mu, x.size)
    cauchy, active = _generalized_cauchy_point(x, g, lb, ub, operator)
    candidate = _subspace_minimization(cauchy, active, x, g, lb, ub, operator)
    direction = candidate - x
    if not np.all(np.isfinite(direction)) or np.dot(g, direction) >= 0.0:
        direction = -projected_gradient(x, g, lb, ub)
    return direction


def qn_ntrqnb(
    prob: BaseProblem,
    bounds: BoundsInput,
    param: NTRQNParameter,
    method: Method,
    callback: Union[Callback, None] = None,
    verbose: bool = False,
) -> tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    """Run projected NTRQN subject to box constraints."""
    prob.reset()
    eps = prob.get_machine_eps()

    lb, ub = prepare_bounds(bounds, prob.n)
    x = np.clip(np.asarray(prob.x0, dtype=np.float64), lb, ub)

    if callback:
        initial_g = prob.g(x, count=False)
        callback.start(
            prob,
            x,
            gnorm_vector=projected_gradient(x, initial_g, lb, ub),
        )

    fx = prob.f(x)
    g = prob.g(x)
    pg = projected_gradient(x, g, lb, ub)

    pg_norm = np.float64(np.linalg.norm(pg))
    if not np.isfinite(fx) or not np.isfinite(pg_norm):
        if callback:
            callback.callback(prob, x, fx, g, gnorm_vector=pg)
        return RetCode.ERR_NUMERICAL_OVERFLOW, fx, x
    if np.linalg.norm(pg, ord=np.inf) <= param.gtol:
        return RetCode.ALREADY_MINIMIZED, fx, x

    memory = QuasiNewtonMemory(pg, param.m, method)
    past_fx: deque[np.float64] = deque([], maxlen=param.past)
    reference_values: deque[np.float64] = deque([fx], maxlen=param.non_monotone)

    k = 0
    mu = np.float64(0.0)
    var_sigma = np.float64(1e-20)
    offo = np.float64(np.sqrt(var_sigma))
    is_offo_mode = False
    min_fx_minus_delta = np.float64(np.inf)
    rejection_counter = 0

    def gradient_mapping(
        point: npt.NDArray[np.float64], gradient: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return projected_gradient(point, gradient, lb, ub)

    while True:
        if min_fx_minus_delta >= fx:
            is_offo_mode = False
            mu = np.float64(0.0)
            if min_fx_minus_delta - fx >= 1.0:
                offo = np.float64(np.sqrt(var_sigma))
        else:
            if not is_offo_mode and verbose:
                print(f"  ⚠️  Switching to offo mode. {k=}")
            is_offo_mode = True
            offo = np.sqrt(offo**2 + pg_norm**2)
            mu = pg_norm * param.mu_scale
            mu = np.clip(mu, param.mu_min_fraction * offo, offo)

        direction = _box_quasi_newton_direction(method, x, g, memory, mu, lb, ub)

        ref_fx = (
            max(reference_values)  # type: ignore[type-var]
            if len(reference_values) > 0
            else fx
        )
        ls_res, new_x, new_f, new_g, delta, rejection_counter = (
            line_search_relaxed_armijo(
                x,
                fx,
                g,
                direction,
                prob,
                param,
                eps,
                ref_fx,
                verbose,
                is_offo_mode,
                rejection_counter,
                gradient_mapping,
            )
        )
        if ls_res != RetCode.SUCCESS:
            if callback:
                callback.callback(prob, x, fx, g, gnorm_vector=pg)
            return ls_res, fx, x

        memory.add_new_data(new_x, new_f, new_g, x, fx, g, callback, eps)

        if mu == 0.0:
            min_fx_minus_delta = np.minimum(min_fx_minus_delta, fx - delta)

        x, fx, g = new_x, new_f, new_g
        k += 1
        pg = projected_gradient(x, g, lb, ub)
        pg_norm = np.float64(np.linalg.norm(pg))

        if verbose:
            print(
                f"iter:{k:04} mu:{mu:.2e} f:{fx:.6e} "
                f"min_f-delta:{min_fx_minus_delta:.6e} |pg|:{pg_norm:.2e}"
            )
        if callback:
            callback.callback(prob, x, fx, g, gnorm_vector=pg)

        result = check_termination(pg, param, fx, past_fx, k, prob.call_f)
        reference_values.append(fx)
        if result is not None:
            return result, fx, x
