from typing import Dict, Union

import numpy as np

from qnlab.parameter import (
    KanzowParameter,
    LineParameter,
    NtqnParameter,
    NTRQNParameter,
    OwlParameter,
)
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn_basic import qn_gradient_descent, qn_newton
from qnlab.solver.qn_kanzow import qn_kanzow
from qnlab.solver.qn_line import qn_line
from qnlab.solver.qn_ntqn import qn_ntqn
from qnlab.solver.qn_ntrqn import qn_ntrqn
from qnlab.solver.qn_owl import qn_owl
from qnlab.solver.qn_scipy import qn_scipy
from qnlab.util.callback import Callback
from qnlab.util.method import Method


def qn(
    prob: BaseProblem,
    method: Method,
    options: Dict[str, Union[np.float64, int]] = {},
    callback: Union[Callback, None] = None,
    verbose: bool = False,
):
    if method.base == "SciPy":
        return qn_scipy(prob, method, options, callback, verbose)

    if method.base == "GradientDescent":
        return qn_gradient_descent(prob, method, options, callback)

    if method.base == "Newton":
        return qn_newton(prob, method, options, callback)

    if method.base == "Line":
        orthantwise_c = options.get("orthantwise_c", 0.0)
        if orthantwise_c != 0.0:
            return qn_owl(prob, OwlParameter(prob.n, options), method, callback)
        else:
            return qn_line(prob, LineParameter(prob.n, options), method, callback)
    elif method.base == "Kanzow" or method.base == "KanzowSec":
        return qn_kanzow(prob, KanzowParameter(prob.n, options), method, callback)
    elif method.base == "NTQN":
        return qn_ntqn(prob, NtqnParameter(prob.n, options), method, callback, verbose)
    elif method.base == "NTRQN":
        return qn_ntrqn(
            prob, NTRQNParameter(prob.n, options), method, callback, verbose
        )
    else:
        raise ValueError(f"Unknown method: {method}. ")
