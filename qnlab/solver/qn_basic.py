from collections.abc import Callable

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import line_search
from scipy.sparse.linalg import LinearOperator, cg

from qnlab.problem.base import BaseProblem
from qnlab.util.callback import Callback
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode

DirectionFn = Callable[
    [np.ndarray, np.ndarray], tuple[np.ndarray | None, RetCode | None]
]


def _perform_line_search(
    prob: BaseProblem,
    x: np.ndarray,
    fx: np.float64,
    g: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray | None, np.float64, np.ndarray]:
    alpha, _, _, fx_new, _, g_new = line_search(prob.f, prob.g, x, d, g, fx)
    if alpha is None:
        return None, fx, g

    x_new = x + alpha * d
    if fx_new is None:
        fx_new = prob.f(x_new)
    gradient_new = (
        prob.g(x_new) if g_new is None else np.asarray(g_new, dtype=np.float64)
    )

    return x_new, np.float64(fx_new), gradient_new


def _run_first_order(
    prob: BaseProblem,
    direction_fn: DirectionFn,
    max_iter: int,
    tol: float,
    callback: Callback | None,
) -> tuple[RetCode, np.float64, np.ndarray]:
    prob.reset()

    x = np.array(prob.x0, dtype=np.float64)
    fx = prob.f(x)
    g = prob.g(x)

    if callback:
        callback.start(prob, x)

    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            if callback:
                callback.callback(prob, x, fx, g)
            return RetCode.SUCCESS, fx, x

        d, error_code = direction_fn(x, g)
        if d is None:
            return error_code if error_code else RetCode.ERR_LOGICERROR, fx, x

        x_new, fx_new, g_new = _perform_line_search(prob, x, fx, g, d)
        if x_new is None:
            return RetCode.ERR_MAXIMUMLINESEARCH, fx, x

        x, fx, g = x_new, fx_new, g_new

        if callback:
            callback.callback(prob, x, fx, g)

    return RetCode.ERR_MAXIMUMITERATION, fx, x


def qn_gradient_descent(
    prob: BaseProblem,
    method: Method,
    options: dict[str, np.float64 | int] | None = None,
    callback: Callback | None = None,
) -> tuple[RetCode, np.float64, np.ndarray]:
    if options is None:
        options = {}
    assert method.base == "GradientDescent"
    max_iter = int(options.get("max_iter", 1000))
    tol = float(options.get("tol", 1e-6))

    return _run_first_order(prob, lambda _x, g: (-g, None), max_iter, tol, callback)


def qn_newton(
    prob: BaseProblem,
    method: Method,
    options: dict[str, np.float64 | int] | None = None,
    callback: Callback | None = None,
) -> tuple[RetCode, np.float64, np.ndarray]:
    if options is None:
        options = {}
    assert method.base == "Newton"
    max_iter = int(options.get("max_iter", 1000))
    tol = float(options.get("tol", 1e-6))
    cg_tol = float(options.get("cg_tol", 1e-8))
    cg_maxiter = int(options.get("cg_maxiter", prob.n * 10))

    def newton_direction(x: np.ndarray, g: np.ndarray):
        uses_hvp = type(prob)._hvp is not BaseProblem._hvp

        if uses_hvp:
            # Prefer Hessian-vector products when available to avoid forming dense Hessians.
            lin_op = LinearOperator(
                dtype=np.float64,
                shape=(prob.n, prob.n),
                matvec=lambda v: prob._hvp(x, v),  # type: ignore
            )
            direction, info = cg(lin_op, -g, rtol=cg_tol, maxiter=cg_maxiter)
            if info == 0:
                return direction, None

        try:
            hess = prob._hessian(x)
        except NotImplementedError:
            return None, RetCode.ERR_INVALIDPARAMETERS

        try:
            direction = np.linalg.solve(hess, -g)
        except LinAlgError:
            return None, RetCode.ERR_LOGICERROR

        return direction, None

    return _run_first_order(prob, newton_direction, max_iter, tol, callback)
