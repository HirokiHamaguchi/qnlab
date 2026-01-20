from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from qnlab.parameter import NtqnParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.external_files import ntqn
from qnlab.util.callback import Callback
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


# terminate once gradient norm is below param.gtol
class StopOptimization(Exception):
    pass


def qn_ntqn(
    prob: BaseProblem,
    param: NtqnParameter,
    method: Method,
    callback: Union[Callback, None] = None,
    verbose: bool = False,
) -> Tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    assert method.base == "NTQN"
    prob.reset()

    if callback:
        callback.start(prob, prob.x0)

    final_xk = np.empty_like(prob.x0, dtype=np.float64)
    final_fk = np.float64(np.nan)

    def _ntqn_callback_impl(x: npt.NDArray[np.float64]) -> None:
        nonlocal final_xk, final_fk

        fx = prob.f(x, count=False)
        gx = prob.g(x, count=False)
        if callback:
            callback.callback(prob, x, fx, gx)
        if np.linalg.norm(gx, ord=np.inf) < param.gtol:
            final_xk[:] = x
            final_fk = fx
            raise StopOptimization()

    def f_func(x: npt.NDArray[np.float64]) -> np.float64:
        return prob.f(x, count=True)

    def g_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return prob.g(x, count=True)

    # Call ntqn.bfgs_e
    try:
        x_k, f_k, _, _, _, flag, _ = ntqn.bfgs_e(
            f_func,
            g_func,
            prob.x0,
            eps_f=prob.get_eps(),
            eps_g=prob.get_noise(),
            callback=_ntqn_callback_impl,
            options={
                "max_iter": param.max_iterations,
                "max_feval": param.max_evaluations,
                "max_geval": param.max_evaluations,
                "tol": param.gtol,
                "qn_hist_size": param.m,  # Use m from parameter
                "terminate": 1,  # optimize until no more progress
                "display": 1 if verbose else 0,
            },
        )
        final_xk[:] = x_k
        final_fk = f_k
    except StopOptimization:
        flag = 0  # Indicate successful convergence

    # Map ntqn flags to RetCode
    ret_code = {
        0: RetCode.SUCCESS,  # Converged to desired gradient tolerance
        1: RetCode.ERR_MAXIMUMITERATION,  # Reached maximum number of iterations
        2: RetCode.ERR_MAXIMUMEVALUATION,  # Reached maximum number of function evaluations
        3: RetCode.ERR_MAXIMUMEVALUATION,  # Reached maximum number of gradient evaluations
        4: RetCode.SUCCESS,  # Reached noise level of the function (consider converged)
        5: RetCode.SUCCESS,  # Reached noise level of the gradient (consider converged)
        6: RetCode.STOP,  # No more progress made
        7: RetCode.ERR_NUMERICAL_INSTABILITY,  # No more progress due to numerical issues
    }.get(flag, RetCode.ERR_UNKNOWNERROR)

    return ret_code, final_fk, final_xk
