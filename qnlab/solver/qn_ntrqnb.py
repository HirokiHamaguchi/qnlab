"""Limited-memory NTRQN solver for box-constrained optimization."""

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Protocol, Sequence, TypeAlias, Union, runtime_checkable

import numpy as np
import numpy.typing as npt

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn_ntrqn import line_search_relaxed_armijo
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
    inverse_coefficients: npt.NDArray[np.float64] | None = None
    vector_gram: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.coefficients.ndim == 1:
            self.coefficients = np.diag(self.coefficients)

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
                np.empty((0, 0), dtype=np.float64),
            )

        workspace = memory.workspace
        steps = workspace.steps
        gradients = workspace.gradients
        step_products = workspace.step_products
        step_gradient = workspace.step_gradient
        gradient_products = workspace.gradient_products
        regularized_pair_products = np.diag(step_gradient) + mu * np.diag(step_products)
        valid = np.isfinite(regularized_pair_products) & (
            regularized_pair_products > 0.0
        )
        if not np.any(valid):
            diagonal = mu if mu > 0.0 else np.float64(1.0 / memory.zero_length)
            return cls(
                np.float64(diagonal),
                np.empty((n, 0), dtype=np.float64),
                np.empty((0, 0), dtype=np.float64),
            )

        if not np.all(valid):
            indices = np.flatnonzero(valid)
            steps = steps[:, indices]
            gradients = gradients[:, indices]
            step_products = step_products[np.ix_(indices, indices)]
            step_gradient = step_gradient[np.ix_(indices, indices)]
            gradient_products = gradient_products[np.ix_(indices, indices)]

        regularized_step_gradient = step_gradient + mu * step_products
        regularized_gradient_products = (
            gradient_products
            + mu * (step_gradient + step_gradient.T)
            + mu * mu * step_products
        )
        last_ys = regularized_step_gradient[-1, -1]
        diagonal = np.float64(regularized_gradient_products[-1, -1] / last_ys)
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            diagonal = mu if mu > 0.0 else np.float64(1.0 / memory.zero_length)
            return cls(
                np.float64(diagonal),
                np.empty((n, 0), dtype=np.float64),
                np.empty((0, 0), dtype=np.float64),
            )

        lower = np.tril(regularized_step_gradient, k=-1)
        pair_products = np.diag(regularized_step_gradient)
        col = pair_products.size
        middle = np.empty((2 * col, 2 * col), dtype=np.float64)
        middle[:col, :col] = diagonal * step_products
        middle[:col, col:] = lower
        middle[col:, :col] = lower.T
        middle[col:, col:] = -np.diag(pair_products)

        vectors = np.concatenate((diagonal * steps, gradients + mu * steps), axis=1)
        vector_gram = np.empty_like(middle)
        vector_gram[:col, :col] = diagonal * diagonal * step_products
        vector_gram[:col, col:] = diagonal * regularized_step_gradient
        vector_gram[col:, :col] = vector_gram[:col, col:].T
        vector_gram[col:, col:] = regularized_gradient_products
        try:
            coefficients = np.linalg.solve(middle, -np.eye(middle.shape[0]))
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                middle, -np.eye(middle.shape[0]), rcond=None
            )[0]
        coefficients = 0.5 * (coefficients + coefficients.T)
        if not np.all(np.isfinite(coefficients)):
            return cls(
                diagonal,
                np.empty((n, 0), dtype=np.float64),
                np.empty((0, 0), dtype=np.float64),
            )
        return cls(
            diagonal,
            vectors,
            coefficients,
            -middle,
            vector_gram,
        )

    def apply(self, vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the compact Hessian approximation."""
        result = self.diagonal * vector
        if self.vectors.shape[1] > 0:
            result = result + self.vectors @ (
                self.coefficients @ (self.vectors.T @ vector)
            )
        return result

    def quadratic_form(self, vector: npt.NDArray[np.float64]) -> np.float64:
        """Return ``vector.T @ B @ vector`` without forming ``B``."""
        compact = self.vectors.T @ vector
        return np.float64(
            self.diagonal * np.dot(vector, vector)
            + np.dot(compact, self.coefficients @ compact)
        )

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
            stored_inverse = self.inverse_coefficients
            if stored_inverse is None:
                try:
                    inverse_coefficients: npt.NDArray[np.float64] = np.asarray(
                        np.linalg.solve(
                            self.coefficients, np.eye(self.coefficients.shape[0])
                        ),
                        dtype=np.float64,
                    )
                except np.linalg.LinAlgError:
                    inverse_coefficients = np.asarray(
                        np.linalg.pinv(self.coefficients), dtype=np.float64
                    )
            else:
                inverse_coefficients = stored_inverse
            middle = inverse_coefficients.copy()
            active = ~free
            if self.vector_gram is not None and np.count_nonzero(
                active
            ) < np.count_nonzero(free):
                active_vectors = self.vectors[active]
                restricted_gram = self.vector_gram - active_vectors.T @ active_vectors
            else:
                restricted_gram = restricted_vectors.T @ restricted_vectors
            middle += restricted_gram / self.diagonal
            middle = 0.5 * (middle + middle.T)
            right = restricted_vectors.T @ restricted_rhs / self.diagonal
            try:
                correction = np.linalg.solve(middle, right)
            except np.linalg.LinAlgError:
                correction = np.linalg.lstsq(middle, right, rcond=None)[0]
            if not np.all(np.isfinite(correction)):
                return result
            solution -= restricted_vectors @ correction / self.diagonal
        result[free] = solution
        return result


@dataclass
class _BoxWorkspace:
    """Reusable arrays whose size depends only on the problem dimension."""

    breakpoint_values: npt.NDArray[np.float64]

    @classmethod
    def create(cls, n: int) -> "_BoxWorkspace":
        return cls(np.empty(n, dtype=np.float64))


def _generalized_cauchy_point(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    operator: _CompactBFGS,
    workspace: _BoxWorkspace | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64],
]:
    """Find the first model minimizer along the projected-gradient path.

    Derivative and curvature data are updated at each breakpoint using the
    compact BFGS representation.  After the initial matrix-vector product,
    each crossed breakpoint therefore costs ``O(m**2)`` rather than
    ``O(n*m)``.
    """
    path_direction = -g.copy()
    active = (lb == ub) | ((x <= lb) & (g > 0.0)) | ((x >= ub) & (g < 0.0))
    path_direction[active] = 0.0
    if not np.any(path_direction):
        return x.copy(), active, np.zeros(operator.vectors.shape[1])

    breakpoint_values = (
        np.full(x.size, np.inf, dtype=np.float64)
        if workspace is None
        else workspace.breakpoint_values
    )
    breakpoint_values.fill(np.inf)
    toward_lower = (path_direction < 0.0) & np.isfinite(lb)
    toward_upper = (path_direction > 0.0) & np.isfinite(ub)
    breakpoint_values[toward_lower] = (
        lb[toward_lower] - x[toward_lower]
    ) / path_direction[toward_lower]
    breakpoint_values[toward_upper] = (
        ub[toward_upper] - x[toward_upper]
    ) / path_direction[toward_upper]
    bounded = np.flatnonzero(
        np.isfinite(breakpoint_values) & (breakpoint_values >= 0.0)
    )
    if bounded.size > 0:
        first_index = int(bounded[np.argmin(breakpoint_values[bounded])])
        first_breakpoint = float(breakpoint_values[first_index])
    else:
        first_index = -1
        first_breakpoint = np.inf
    first_pending = True
    breakpoints: list[tuple[float, int]] | None = None

    compact_direction = operator.vectors.T @ path_direction
    compact_step = np.zeros(operator.vectors.shape[1], dtype=np.float64)
    derivative = np.float64(np.dot(g, path_direction))
    curvature = operator.quadratic_form(path_direction)
    if not np.isfinite(curvature) or curvature <= 0.0:
        return x.copy(), active, compact_step
    original_curvature = curvature
    curvature_floor = np.finfo(np.float64).eps * original_curvature
    elapsed = np.float64(0.0)

    while True:
        if first_pending:
            breakpoint, index = first_breakpoint, first_index
            first_pending = False
        else:
            if breakpoints is None:
                breakpoints = [
                    (float(breakpoint_values[i]), int(i))
                    for i in bounded
                    if i != first_index
                ]
                heapq.heapify(breakpoints)
            breakpoint, index = (
                heapq.heappop(breakpoints) if breakpoints else (np.inf, -1)
            )
        interval = np.float64(max(0.0, breakpoint - elapsed))
        minimizer = np.float64(-derivative / curvature)
        if minimizer < interval:
            minimizer = np.maximum(0.0, minimizer)
            elapsed += minimizer
            compact_step += minimizer * compact_direction
            return np.clip(x - elapsed * g, lb, ub), active, compact_step

        compact_step += interval * compact_direction
        derivative += interval * curvature
        elapsed = np.float64(breakpoint)

        old_component = path_direction[index]
        bound_step = (
            ub[index] - x[index] if old_component > 0.0 else lb[index] - x[index]
        )
        row = operator.vectors[index]
        model_gradient_component = (
            g[index]
            + operator.diagonal * bound_step
            + np.dot(row, operator.coefficients @ compact_step)
        )
        derivative -= old_component * model_gradient_component

        matrix_direction_component = operator.diagonal * old_component + np.dot(
            row, operator.coefficients @ compact_direction
        )
        matrix_diagonal = operator.diagonal + np.dot(row, operator.coefficients @ row)
        curvature += (
            -2.0 * old_component * matrix_direction_component
            + old_component * old_component * matrix_diagonal
        )
        curvature = np.float64(max(curvature_floor, curvature))
        compact_direction -= old_component * row
        path_direction[index] = 0.0
        active[index] = True


def _subspace_minimization(
    cauchy: npt.NDArray[np.float64],
    active: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    operator: _CompactBFGS,
    compact_step: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Take one safeguarded Newton correction in the Cauchy free subspace."""
    free = ~active
    if not np.any(free):
        return cauchy

    step = cauchy - x
    if compact_step is None:
        compact_step = operator.vectors.T @ step
    model_gradient = g + operator.diagonal * step
    if operator.vectors.shape[1] > 0:
        model_gradient += operator.vectors @ (operator.coefficients @ compact_step)
    correction = operator.solve_restricted(-model_gradient, free)
    if not np.all(np.isfinite(correction)) or np.dot(model_gradient, correction) >= 0.0:
        return cauchy

    alpha = min(1.0, float(max_feasible_step(cauchy, correction, lb, ub)))
    if alpha <= 0.0:
        return cauchy
    return np.clip(cauchy + alpha * correction, lb, ub)


def _box_quasi_newton_direction(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    memory: QuasiNewtonMemory,
    mu: np.float64,
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
    workspace: _BoxWorkspace | None = None,
) -> npt.NDArray[np.float64]:
    """Approximately minimize the regularized L-BFGS model over the box."""
    operator = _CompactBFGS.from_memory(memory, mu, x.size)
    cauchy, active, compact_step = _generalized_cauchy_point(
        x, g, lb, ub, operator, workspace
    )
    candidate = _subspace_minimization(
        cauchy, active, x, g, lb, ub, operator, compact_step
    )
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
    if method.update != "bfgs":
        raise ValueError("NTRQNB supports only the BFGS update")
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
    box_workspace = _BoxWorkspace.create(prob.n)
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

        direction = _box_quasi_newton_direction(x, g, memory, mu, lb, ub, box_workspace)

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
