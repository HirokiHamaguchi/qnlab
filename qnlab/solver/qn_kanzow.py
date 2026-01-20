from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

import qnlab.solver.external_files.kanzow.utility.parameters as kanzow
from qnlab.parameter import KanzowParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.external_files.kanzow import regLBFGS
from qnlab.util.callback import Callback
from qnlab.util.method import Method
from qnlab.util.ret_values import RetCode


def qn_kanzow(
    prob: BaseProblem,
    param: KanzowParameter,
    method: Method,
    callback: Union[Callback, None] = None,
) -> Tuple[RetCode, np.float64, npt.NDArray[np.float64]]:
    """
    Wrapper for the Kanzow regularization method.
    This wrapper adapts the kanzow.solveNonmonotone function to work with the qnlab interface.
    """
    assert method.base == "Kanzow"
    prob.reset()

    # Configure kanzow parameters from param
    original_maxIter = kanzow.maxIter
    original_maxEval = kanzow.maxEval
    original_tolGrad = kanzow.tolGrad
    original_memory = kanzow.memory

    def f_counted(x: np.ndarray):
        return prob.f(x, count=True)

    def g_with_callback(x: np.ndarray):
        g = prob.g(x)
        if callback:
            callback.callback(prob, x, prob.f(x, count=False), g)
        return g

    try:
        # Set parameters from param
        kanzow.maxIter = param.max_iterations
        kanzow.maxEval = param.max_evaluations
        kanzow.tolGrad = float(param.gtol)
        kanzow.memory = param.m

        result = regLBFGS.solveNonmonotone(f_counted, g_with_callback, prob.x0)

        x_opt = result[0]
        iter_info = result[1]  # [successful_iters, total_evals]
        fx = prob.f(x_opt, count=False)
        g = prob.g(x_opt, count=False)

        # Determine return code based on iteration info
        successful_iters = iter_info[0] if len(iter_info) > 0 else 0
        total_evals = iter_info[1] if len(iter_info) > 1 else 0

        gnorm = np.linalg.norm(g, np.inf)

        if gnorm <= kanzow.tolGrad:
            ret_code = RetCode.SUCCESS
        elif successful_iters >= kanzow.maxIter:
            ret_code = RetCode.ERR_MAXIMUMITERATION
        elif total_evals >= kanzow.maxEval:
            ret_code = RetCode.ERR_MAXIMUMEVALUATION
        else:
            ret_code = RetCode.ERR_UNKNOWNERROR

        return ret_code, np.float64(fx), x_opt

    finally:
        # Restore original kanzow parameters
        kanzow.maxIter = original_maxIter
        kanzow.maxEval = original_maxEval
        kanzow.tolGrad = original_tolGrad
        kanzow.memory = original_memory
