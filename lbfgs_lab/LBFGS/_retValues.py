# Return values of lbfgs().
# Roughly speaking, a negative value indicates an error.

LBFGS_SUCCESS = 0
LBFGS_CONVERGENCE = 0
LBFGS_STOP = 1
LBFGS_ALREADY_MINIMIZED = 2

LBFGSERR_UNKNOWNERROR = -1024
LBFGSERR_LOGICERROR = -1023
LBFGSERR_OUTOFMEMORY = -1022
LBFGSERR_CANCELED = -1021
LBFGSERR_INVALID_N = -1020
LBFGSERR_INVALID_N_SSE = -1019
LBFGSERR_INVALID_X_SSE = -1018
LBFGSERR_INVALID_EPSILON = -1017
LBFGSERR_INVALID_TESTPERIOD = -1016
LBFGSERR_INVALID_DELTA = -1015
LBFGSERR_INVALID_LINESEARCH = -1014
LBFGSERR_INVALID_MINSTEP = -1013
LBFGSERR_INVALID_MAXSTEP = -1012
LBFGSERR_INVALID_FTOL = -1011
LBFGSERR_INVALID_WOLFE = -1010
LBFGSERR_INVALID_GTOL = -1009
LBFGSERR_INVALID_XTOL = -1008
LBFGSERR_INVALID_MAXLINESEARCH = -1007
LBFGSERR_INVALID_ORTHANTWISE = -1006
LBFGSERR_INVALID_ORTHANTWISE_START = -1005
LBFGSERR_INVALID_ORTHANTWISE_END = -1004
LBFGSERR_OUTOFINTERVAL = -1003
LBFGSERR_INCORRECT_TMINMAX = -1002
LBFGSERR_ROUNDING_ERROR = -1001
LBFGSERR_MINIMUMSTEP = -1000
LBFGSERR_MAXIMUMSTEP = -999
LBFGSERR_MAXIMUMLINESEARCH = -998
LBFGSERR_MAXIMUMITERATION = -997
LBFGSERR_WIDTHTOOSMALL = -996
LBFGSERR_INVALIDPARAMETERS = -995
LBFGSERR_INCREASEGRADIENT = -994


def lbfgs_strerror(err: int):
    """
    Returns a string representation of the given L-BFGS error code.
    """
    error_messages = {
        # Also handles LBFGS_CONVERGENCE.
        LBFGS_SUCCESS: "Success: reached convergence (gtol).",
        LBFGS_STOP: "Success: met stopping criteria (ftol).",
        LBFGS_ALREADY_MINIMIZED: "The initial variables already minimize the objective function.",
        LBFGSERR_UNKNOWNERROR: "Unknown error.",
        LBFGSERR_LOGICERROR: "Logic error.",
        LBFGSERR_OUTOFMEMORY: "Insufficient memory.",
        LBFGSERR_CANCELED: "The minimization process has been canceled.",
        LBFGSERR_INVALID_N: "Invalid number of variables specified.",
        LBFGSERR_INVALID_N_SSE: "Invalid number of variables (for SSE) specified.",
        LBFGSERR_INVALID_X_SSE: "The array x must be aligned to 16 (for SSE).",
        LBFGSERR_INVALID_EPSILON: "Invalid parameter lbfgs_parameter_t::epsilon specified.",
        LBFGSERR_INVALID_TESTPERIOD: "Invalid parameter lbfgs_parameter_t::past specified.",
        LBFGSERR_INVALID_DELTA: "Invalid parameter lbfgs_parameter_t::delta specified.",
        LBFGSERR_INVALID_LINESEARCH: "Invalid parameter lbfgs_parameter_t::linesearch specified.",
        LBFGSERR_INVALID_MINSTEP: "Invalid parameter lbfgs_parameter_t::max_step specified.",
        LBFGSERR_INVALID_MAXSTEP: "Invalid parameter lbfgs_parameter_t::max_step specified.",
        LBFGSERR_INVALID_FTOL: "Invalid parameter lbfgs_parameter_t::ftol specified.",
        LBFGSERR_INVALID_WOLFE: "Invalid parameter lbfgs_parameter_t::wolfe specified.",
        LBFGSERR_INVALID_GTOL: "Invalid parameter lbfgs_parameter_t::gtol specified.",
        LBFGSERR_INVALID_XTOL: "Invalid parameter lbfgs_parameter_t::xtol specified.",
        LBFGSERR_INVALID_MAXLINESEARCH: "Invalid parameter lbfgs_parameter_t::max_linesearch specified.",
        LBFGSERR_INVALID_ORTHANTWISE: "Invalid parameter lbfgs_parameter_t::orthantwise_c specified.",
        LBFGSERR_INVALID_ORTHANTWISE_START: "Invalid parameter lbfgs_parameter_t::orthantwise_start specified.",
        LBFGSERR_INVALID_ORTHANTWISE_END: "Invalid parameter lbfgs_parameter_t::orthantwise_end specified.",
        LBFGSERR_OUTOFINTERVAL: "The line-search step went out of the interval of uncertainty.",
        LBFGSERR_INCORRECT_TMINMAX: "A logic error occurred; alternatively, the interval of uncertainty became too small.",
        LBFGSERR_ROUNDING_ERROR: "A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions.",
        LBFGSERR_MINIMUMSTEP: "The line-search step became smaller than lbfgs_parameter_t::min_step.",
        LBFGSERR_MAXIMUMSTEP: "The line-search step became larger than lbfgs_parameter_t::max_step.",
        LBFGSERR_MAXIMUMLINESEARCH: "The line-search routine reaches the maximum number of evaluations.",
        LBFGSERR_MAXIMUMITERATION: "The algorithm routine reaches the maximum number of iterations.",
        LBFGSERR_WIDTHTOOSMALL: "Relative width of the interval of uncertainty is at most lbfgs_parameter_t::xtol.",
        LBFGSERR_INVALIDPARAMETERS: "A logic error (negative line-search step) occurred.",
        LBFGSERR_INCREASEGRADIENT: "The current search direction increases the objective function value.",
    }
    return error_messages.get(err, "(unknown)")
