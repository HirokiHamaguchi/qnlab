from collections import deque
from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize._linesearch import _cubicmin, _quadmin  # type: ignore

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.update.update import get_direction_reg
from qnlab.util.callback import Callback
from qnlab.util.check_termination import check_termination
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


def line_search_relaxed_armijo(
    x: npt.NDArray[np.float64],
    fx: np.float64,
    g: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    prob: BaseProblem,
    param: NTRQNParameter,
    eps: np.float64,
    ref_fx: np.float64,
    verbose: bool,
    is_offo_mode: bool,
    rejection_counter: int,
    gradient_mapping: Union[
        Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
        ],
        None,
    ] = None,
) -> Tuple[
    RetCode,
    npt.NDArray[np.float64],
    np.float64,
    npt.NDArray[np.float64],
    np.float64,
    int,
]:
    alpha: np.float64 = np.float64(1.0)

    # delta1/delta2 mirror SciPy's _zoom heuristics:
    # keep cubic/quadratic interpolants away from the bracket endpoints for robustness.
    delta1 = np.float64(0.2)  # cubic interpolant check
    delta2 = np.float64(0.1)  # quadratic interpolant check

    prev_alpha: Union[np.float64, None] = None
    prev_f: Union[np.float64, None] = None

    dg = np.dot(d, g)
    assert dg < 0, "Search direction d is not a descent direction."

    for _ in range(param.max_linesearch):
        x_try = x + alpha * d

        f_try = prob.f(x_try)

        if not np.isfinite(f_try):
            if verbose:
                print("  ⚠️  Rejected due to Inf/NaN in function value.")
            alpha *= np.float64(0.5)
            continue

        if np.all(x_try == x):
            if verbose:
                print("  ⚠️ Rejected due to no change in x.")
            return (
                RetCode.ERR_NUMERICAL_INSTABILITY,
                x_try,
                f_try,
                prob.g(x_try),
                np.float64(np.inf),
                rejection_counter,
            )

        # relaxed Armijo condition (non-monotone reference)
        delta = 2 * eps / (1 - eps) * max(fx, -f_try, 1.0)
        if ref_fx + param.armijo * alpha * dg + delta < f_try:
            # Armijo failed: try to shrink alpha using SciPy _zoom-style logic.
            old_alpha = alpha
            best_alpha: Union[np.float64, None] = None

            if prev_alpha is not None:
                # Try cubic interpolation (_cubicmin) if possible.
                cchk = delta1 * old_alpha
                cand = _cubicmin(
                    np.float64(0.0), fx, dg, old_alpha, f_try, prev_alpha, prev_f
                )
                if cand is not None and cchk < cand < old_alpha - cchk:
                    best_alpha = cand

            if best_alpha is None:
                # Fall back to quadratic interpolation (_quadmin)
                qchk = delta2 * old_alpha
                cand = _quadmin(np.float64(0.0), fx, dg, old_alpha, f_try)
                if (
                    cand is not None
                    and np.isfinite(cand)
                    and qchk < cand < old_alpha - qchk
                ):
                    best_alpha = cand

            if best_alpha is None or best_alpha > old_alpha:
                best_alpha = np.float64(0.5) * old_alpha

            alpha = np.float64(best_alpha)
            prev_alpha, prev_f = old_alpha, f_try
            continue

        g_try = prob.g(x_try)
        if not np.all(np.isfinite(g_try)):
            if verbose:
                print("  ⚠️  Rejected due to Inf/NaN in gradient value.")
            alpha /= 4
            rejection_counter += 1
            if rejection_counter >= param.max_inf_nan_rejections:
                return (
                    RetCode.ERR_NUMERICAL_INSTABILITY,
                    x_try,
                    f_try,
                    g_try,
                    delta,
                    rejection_counter,
                )
            continue

        stationarity = (
            g_try if gradient_mapping is None else gradient_mapping(x_try, g_try)
        )
        g_try_norm = np.float64(np.linalg.norm(stationarity))
        if is_offo_mode and alpha == 1.0 and g_try_norm > param.gtol:
            dg_try = np.dot(d, g_try)
            dnorm = np.float64(np.linalg.norm(d))
            assert dg < 0, "Search direction d is not a descent direction."
            if dg_try > 0.5 * dnorm * g_try_norm and dg < 0:
                alpha *= np.clip(1 / 16, -dg / (dg_try - dg), 15 / 16)
                continue

        return RetCode.SUCCESS, x_try, f_try, g_try, delta, rejection_counter
    else:
        if verbose:
            print("  ❌ Line search failed: maximum number of iterations exceeded.")
        return (
            RetCode.ERR_MAXIMUMLINESEARCH,
            x,
            fx,
            g,
            np.float64(np.inf),
            rejection_counter,
        )


def qn_ntrqn(
    prob: BaseProblem,
    param: NTRQNParameter,
    method: Method,
    callback: Union[Callback, None] = None,
    verbose: bool = False,
) -> Tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    """Runs the L-BFGS algorithm for unconstrained optimization."""
    prob.reset()
    eps = prob.get_machine_eps()

    x = np.array(prob.x0, dtype=np.float64)

    if callback:
        callback.start(prob, x)

    fx = prob.f(x)
    g = prob.g(x)

    lm = QuasiNewtonMemory(g, param.m, method)
    pf: deque[np.float64] = deque([], maxlen=param.past)
    pf2: deque[np.float64] = deque([fx], maxlen=param.non_monotone)
    gnorm: np.float64 = np.float64(np.linalg.norm(g))

    if not np.isfinite(gnorm):
        if callback:
            callback.callback(prob, x, fx, g)
        return RetCode.ERR_NUMERICAL_OVERFLOW, fx, x

    k = 0
    mu = np.float64(0.0)
    var_sigma = np.float64(1e-20)
    offo = np.float64(np.sqrt(var_sigma))
    is_offo_mode = False
    min_fx_minus_delta = np.float64(np.inf)
    rejection_counter: int = 0

    while True:
        if min_fx_minus_delta >= fx:
            is_offo_mode = False
            mu = np.float64(0.0)
            # if sufficient decrease is observed, reset offo
            if min_fx_minus_delta - fx >= 1.0:
                offo = np.float64(np.sqrt(var_sigma))
        else:
            if not is_offo_mode and verbose:
                print(f"  ⚠️  Switching to offo mode. {k=}")
            is_offo_mode = True
            offo = np.sqrt(offo**2 + gnorm**2)
            mu = gnorm * param.mu_scale
            mu = np.clip(mu, param.mu_min_fraction * offo, offo)

        d = get_direction_reg(method, x, g, lm, mu)

        ref_fx = max(pf2) if len(pf2) > 0 else fx  # type: ignore[type-var]
        ls_res, new_x, new_f, new_g, delta, rejection_counter = (
            line_search_relaxed_armijo(
                x,
                fx,
                g,
                d,
                prob,
                param,
                eps,
                ref_fx,
                verbose,
                is_offo_mode,
                rejection_counter,
            )
        )
        if ls_res != RetCode.SUCCESS:
            if callback:
                callback.callback(prob, x, fx, g)
            return ls_res, fx, x

        lm.add_new_data(new_x, new_f, new_g, x, fx, g, callback, eps)

        if mu == 0.0:
            min_fx_minus_delta = np.minimum(min_fx_minus_delta, fx - delta)

        x, fx, g = new_x, new_f, new_g
        k += 1
        gnorm = np.float64(np.linalg.norm(g))

        if verbose:
            print(
                f"iter:{k:04} mu:{mu:.2e} f:{fx:.6e} min_f-delta:{min_fx_minus_delta:.6e} "
            )
        if callback:
            callback.callback(prob, x, fx, g)

        result = check_termination(g, param, fx, pf, k, prob.call_f)
        pf2.append(fx)
        if result is not None:
            return result, fx, x
