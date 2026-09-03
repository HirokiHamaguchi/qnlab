from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.optimize

from qnlab.problem.base import BaseProblem
from qnlab.solver.qn_ntrqnb import (
    BoundsInput,
    prepare_bounds,
    projected_gradient,
)
from qnlab.util.callback import Callback
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


def qn_scipy(
    prob: BaseProblem,
    method: Method,
    option: dict | None = None,
    callback: Callback | None = None,
    verbose: bool = False,
    bounds: BoundsInput | None = None,
):
    if option is None:
        option = {}
    prob.reset()
    assert method.scipy_method != "None"

    def objective(x: npt.NDArray[np.float64]) -> np.float64:
        return prob.f(x)

    def gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return prob.g(x)

    def hessian_product(
        x: npt.NDArray[np.float64], p: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return prob.hvp(x, p)

    gradient_mapping: (
        Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ]
        | None
    ) = None
    scipy_bounds = None
    if bounds is not None:
        lb, ub = prepare_bounds(bounds, prob.n)
        scipy_bounds = scipy.optimize.Bounds(lb, ub)

        def map_gradient(
            x: npt.NDArray[np.float64], g: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return projected_gradient(x, g, lb, ub)

        gradient_mapping = map_gradient

    scipy_options = cast(Any, option)
    last_recorded_x: npt.NDArray[np.float64] | None = None

    def record_callback(xk: npt.NDArray[np.float64]) -> None:
        nonlocal last_recorded_x
        assert callback is not None
        if last_recorded_x is not None and np.array_equal(xk, last_recorded_x):
            return
        fx = prob.f(xk, count=False)
        g = prob.g(xk, count=False)
        stationarity = None if gradient_mapping is None else gradient_mapping(xk, g)
        callback.callback(prob, xk, fx, g, gnorm_vector=stationarity)
        last_recorded_x = np.copy(xk)

    def callback_scipy_func(xk: npt.NDArray[np.float64]) -> None:
        record_callback(xk)

    def callback_scipy_result(
        intermediate_result: scipy.optimize.OptimizeResult,
    ) -> None:
        record_callback(np.asarray(intermediate_result.x, dtype=np.float64))

    if callback:
        initial_stationarity = None
        if gradient_mapping is not None:
            initial_g = prob.g(prob.x0, count=False)
            initial_stationarity = gradient_mapping(prob.x0, initial_g)
        callback.start(prob, prob.x0, gnorm_vector=initial_stationarity)
        last_recorded_x = np.copy(prob.x0)

    if method.scipy_method in ["Powell", "Nelder-Mead", "COBYLA"]:
        res = scipy.optimize.minimize(
            objective,
            prob.x0,
            method=method.scipy_method,
            callback=callback_scipy_func if callback else None,
            options=scipy_options,
            bounds=scipy_bounds,
        )
    elif method.scipy_method == "trust-constr":
        res = scipy.optimize.minimize(
            objective,
            prob.x0,
            jac=gradient,
            method=method.scipy_method,
            callback=callback_scipy_result if callback else None,
            options=scipy_options,
            hessp=hessian_product,  # trust-constr can be used with None
            bounds=scipy_bounds,
        )
    elif method.scipy_method in ["Newton-CG", "trust-ncg", "trust-krylov"]:
        res = scipy.optimize.minimize(
            objective,
            prob.x0,
            jac=gradient,
            method=method.scipy_method,
            callback=callback_scipy_func if callback else None,
            options=scipy_options,
            hessp=hessian_product,
            bounds=scipy_bounds,
        )
    else:
        res = scipy.optimize.minimize(
            objective,
            prob.x0,
            jac=gradient,
            method=method.scipy_method,
            callback=callback_scipy_func if callback else None,
            options=scipy_options,
            bounds=scipy_bounds,
        )

    if verbose:
        print(f"{res.message=}")

    if callback:
        record_callback(np.asarray(res.x, dtype=np.float64))

    return RetCode.SUCCESS if res.success else RetCode.ERR_UNKNOWNERROR, res.fun, res.x
