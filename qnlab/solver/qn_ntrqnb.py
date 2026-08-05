"""Limited-memory NTRQN solver for box-constrained optimization."""

from collections import deque
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


def _projected_direction(
    method: Method,
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    memory: QuasiNewtonMemory,
    mu: np.float64,
    lb: npt.NDArray[np.float64],
    ub: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Project the NTRQN step, safeguarding it by projected-gradient descent."""
    ntrqn_direction = get_direction_reg(method, x, g, memory, mu)
    direction = np.clip(ntrqn_direction, lb - x, ub - x)
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

    memory = QuasiNewtonMemory(g, param.m, method)
    past_fx: deque[np.float64] = deque([], maxlen=param.past)
    reference_values: deque[np.float64] = deque([fx], maxlen=param.non_monotone)

    k = 0
    mu = np.float64(0.0)
    var_sigma = np.float64(1e-10)
    offo: np.float64 = var_sigma
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
                offo = var_sigma
        else:
            if not is_offo_mode and verbose:
                print(f"  ⚠️  Switching to offo mode. {k=}")
            is_offo_mode = True
            offo = np.sqrt(offo**2 + pg_norm**2)
            mu = pg_norm * param.mu_scale
            mu = np.clip(mu, param.mu_min_fraction * offo, offo)

        direction = _projected_direction(method, x, g, memory, mu, lb, ub)

        ref_fx = max(reference_values) if len(reference_values) > 0 else fx
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
