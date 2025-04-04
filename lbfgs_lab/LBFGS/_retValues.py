from enum import IntEnum, auto


class _NegativeAutoEnum(IntEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        if not last_values:
            return -1
        return last_values[-1] - 1


class RetCode(_NegativeAutoEnum):
    # Return values of lbfgs().
    # Roughly speaking, a negative value indicates an error.
    SUCCESS = 0
    STOP = 1
    ALREADY_MINIMIZED = 2
    LINESEARCH_SUCCESS = 3
    ERR_UNKNOWNERROR = auto()
    ERR_LOGICERROR = auto()
    ERR_OUTOFMEMORY = auto()
    ERR_CANCELED = auto()
    ERR_INVALID_N = auto()
    ERR_INVALID_M = auto()
    ERR_INVALID_N_SSE = auto()
    ERR_INVALID_X_SSE = auto()
    ERR_INVALID_EPSILON = auto()
    ERR_INVALID_TESTPERIOD = auto()
    ERR_INVALID_DELTA = auto()
    ERR_INVALID_LINESEARCH = auto()
    ERR_INVALID_MINSTEP = auto()
    ERR_INVALID_MAXSTEP = auto()
    ERR_INVALID_FTOL = auto()
    ERR_INVALID_WOLFE = auto()
    ERR_INVALID_GTOL = auto()
    ERR_INVALID_XTOL = auto()
    ERR_INVALID_MAXLINESEARCH = auto()
    ERR_INVALID_ORTHANTWISE = auto()
    ERR_INVALID_ORTHANTWISE_START = auto()
    ERR_INVALID_ORTHANTWISE_END = auto()
    ERR_OUTOFINTERVAL = auto()
    ERR_INCORRECT_TMINMAX = auto()
    ERR_ROUNDING_ERROR = auto()
    ERR_MINIMUMSTEP = auto()
    ERR_MAXIMUMSTEP = auto()
    ERR_MAXIMUMLINESEARCH = auto()
    ERR_MAXIMUMITERATION = auto()
    ERR_WIDTHTOOSMALL = auto()
    ERR_INVALIDPARAMETERS = auto()
    ERR_INCREASEGRADIENT = auto()

    def __str__(self):
        error_messages = {
            RetCode.SUCCESS: "Success: reached convergence (gtol).",
            RetCode.STOP: "Success: met stopping criteria (ftol).",
            RetCode.ALREADY_MINIMIZED: "The initial variables already minimize the objective function.",
            RetCode.LINESEARCH_SUCCESS: "Line search succeeded.",
            RetCode.ERR_UNKNOWNERROR: "Unknown error.",
            RetCode.ERR_LOGICERROR: "Logic error.",
            RetCode.ERR_OUTOFMEMORY: "Insufficient memory.",
            RetCode.ERR_CANCELED: "The minimization process has been canceled.",
            RetCode.ERR_INVALID_N: "Invalid number of variables specified.",
            RetCode.ERR_INVALID_M: "Invalid number of corrections specified.",
            RetCode.ERR_INVALID_N_SSE: "Invalid number of variables (for SSE) specified.",
            RetCode.ERR_INVALID_X_SSE: "The array x must be aligned to 16 (for SSE).",
            RetCode.ERR_INVALID_EPSILON: "Invalid parameter LBFGSParameter::epsilon specified.",
            RetCode.ERR_INVALID_TESTPERIOD: "Invalid parameter LBFGSParameter::past specified.",
            RetCode.ERR_INVALID_DELTA: "Invalid parameter LBFGSParameter::delta specified.",
            RetCode.ERR_INVALID_LINESEARCH: "Invalid parameter LBFGSParameter::linesearch specified.",
            RetCode.ERR_INVALID_MINSTEP: "Invalid parameter LBFGSParameter::max_step specified.",
            RetCode.ERR_INVALID_MAXSTEP: "Invalid parameter LBFGSParameter::max_step specified.",
            RetCode.ERR_INVALID_FTOL: "Invalid parameter LBFGSParameter::ftol specified.",
            RetCode.ERR_INVALID_WOLFE: "Invalid parameter LBFGSParameter::wolfe specified.",
            RetCode.ERR_INVALID_GTOL: "Invalid parameter LBFGSParameter::gtol specified.",
            RetCode.ERR_INVALID_XTOL: "Invalid parameter LBFGSParameter::xtol specified.",
            RetCode.ERR_INVALID_MAXLINESEARCH: "Invalid parameter LBFGSParameter::max_linesearch specified.",
            RetCode.ERR_INVALID_ORTHANTWISE: "Invalid parameter LBFGSParameter::orthantwise_c specified.",
            RetCode.ERR_INVALID_ORTHANTWISE_START: "Invalid parameter LBFGSParameter::orthantwise_start specified.",
            RetCode.ERR_INVALID_ORTHANTWISE_END: "Invalid parameter LBFGSParameter::orthantwise_end specified.",
            RetCode.ERR_OUTOFINTERVAL: "The line-search step went out of the interval of uncertainty.",
            RetCode.ERR_INCORRECT_TMINMAX: "A logic error occurred; alternatively, the interval of uncertainty became too small.",
            RetCode.ERR_ROUNDING_ERROR: "A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions.",
            RetCode.ERR_MINIMUMSTEP: "The line-search step became smaller than LBFGSParameter::min_step.",
            RetCode.ERR_MAXIMUMSTEP: "The line-search step became larger than LBFGSParameter::max_step.",
            RetCode.ERR_MAXIMUMLINESEARCH: "The line-search routine reaches the maximum number of evaluations.",
            RetCode.ERR_MAXIMUMITERATION: "The algorithm routine reaches the maximum number of iterations.",
            RetCode.ERR_WIDTHTOOSMALL: "Relative width of the interval of uncertainty is at most LBFGSParameter::xtol.",
            RetCode.ERR_INVALIDPARAMETERS: "A logic error (negative line-search step) occurred.",
            RetCode.ERR_INCREASEGRADIENT: "The current search direction increases the objective function value.",
        }
        return error_messages.get(self, "(unknown)")
