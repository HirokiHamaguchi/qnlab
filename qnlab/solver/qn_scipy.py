from collections.abc import Callable

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

    gradient_mapping: (
        Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ]
        | None
    ) = None
    if bounds is not None:
        lb, ub = prepare_bounds(bounds, prob.n)

        def map_gradient(
            x: npt.NDArray[np.float64], g: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            return projected_gradient(x, g, lb, ub)

        gradient_mapping = map_gradient

    def record_callback(xk):
        assert callback is not None
        fx = prob.f(xk, count=False)
        g = prob.g(xk, count=False)
        stationarity = None if gradient_mapping is None else gradient_mapping(xk, g)
        callback.callback(prob, xk, fx, g, gnorm_vector=stationarity)

    def callback_scipy_func(xk):
        record_callback(xk)

    def callback_trust_constr(xk, _):
        record_callback(xk)

    if callback:
        initial_stationarity = None
        if gradient_mapping is not None:
            initial_g = prob.g(prob.x0, count=False)
            initial_stationarity = gradient_mapping(prob.x0, initial_g)
        callback.start(prob, prob.x0, gnorm_vector=initial_stationarity)

    bounds_kwarg = {} if bounds is None else {"bounds": bounds}

    if method.scipy_method in ["Powell", "Nelder-Mead", "COBYLA"]:
        res = scipy.optimize.minimize(
            prob.f,
            prob.x0,
            method=method.scipy_method,
            callback=callback_scipy_func if callback else None,
            options=option,
            **bounds_kwarg,
        )
    elif method.scipy_method in [
        "Newton-CG",
        "trust-constr",
        "trust-ncg",
        "trust-krylov",
    ]:
        cb = (
            callback_trust_constr
            if "trust-constr" in method.scipy_method
            else callback_scipy_func
        )
        res = scipy.optimize.minimize(
            prob.f,
            prob.x0,
            jac=prob.g,
            method=method.scipy_method,
            callback=cb if callback else None,
            options=option,
            hessp=lambda x, p: prob.hvp(x, p),  # trust-constr can be used with None
            **bounds_kwarg,
        )
    else:
        res = scipy.optimize.minimize(
            prob.f,
            prob.x0,
            jac=prob.g,
            method=method.scipy_method,
            callback=callback_scipy_func if callback else None,
            options=option,
            **bounds_kwarg,
        )

    if verbose:
        print(f"{res.message=}")

    return RetCode.SUCCESS if res.success else RetCode.ERR_UNKNOWNERROR, res.fun, res.x
