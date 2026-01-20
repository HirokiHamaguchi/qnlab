from typing import Union

import scipy.optimize

from qnlab.problem.base import BaseProblem
from qnlab.util.callback import Callback
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


def qn_scipy(
    prob: BaseProblem,
    method: Method,
    option: dict = {},
    callback: Union[Callback, None] = None,
    verbose: bool = False,
):
    prob.reset()

    def callback_scipy_func(xk):
        if callback:
            callback.callback(
                prob, xk, prob.f(xk, count=False), prob.g(xk, count=False)
            )

    def callback_trust_constr(xk, _):
        if callback:
            callback.callback(
                prob, xk, prob.f(xk, count=False), prob.g(xk, count=False)
            )

    if callback:
        callback.start(prob, prob.x0)

    if method.scipy_method in ["Powell", "Nelder-Mead", "COBYLA"]:
        res = scipy.optimize.minimize(
            prob.f,
            prob.x0,
            method=method.scipy_method,
            callback=callback_scipy_func,
            options=option,
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
            callback=cb,
            options=option,
            hessp=lambda x, p: prob.hvp(x, p),  # trust-constr can be used with None
        )
    else:
        res = scipy.optimize.minimize(
            prob.f,
            prob.x0,
            jac=prob.g,
            method=method.scipy_method,
            callback=callback_scipy_func,
            options=option,
        )

    if verbose:
        print(f"{res.message=}")

    return RetCode.SUCCESS if res.success else RetCode.ERR_UNKNOWNERROR, res.fun, res.x
