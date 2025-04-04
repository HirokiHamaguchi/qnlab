import math
import numpy as np
import numpy.typing as npt
from typing import Tuple

from ._params import LBFGSParameter
from ._retValues import RetCode
from ._callback import CallbackData


# default (MoreThuente)
LBFGS_LINESEARCH_DEFAULT = 0
# MoreThuente method proposed by More and Thuente
LBFGS_LINESEARCH_MORETHUENTE = 0
# Backtracking method with the Armijo condition:
# f(x + a * d) <= f(x) + lbfgs_parameter_t::ftol * a * g(x)^T d,
# where x is the current point, d is the current search direction,
# and a is the step length.
LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1
# The backtracking method with the default (regular Wolfe) condition.
LBFGS_LINESEARCH_BACKTRACKING = 2
# Backtracking method with regular Wolfe condition.
# both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
# and the curvature condition:
# g(x + a * d)^T d >= lbfgs_parameter_t::wolfe * g(x)^T d,
# where x is the current point, d is the current search direction,
# and a is the step length.
LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2
# Backtracking method with strong Wolfe condition.
# both the Armijo condition (LBFGS_LINESEARCH_BACKTRACKING_ARMIJO)
# and the following condition:
# |g(x + a * d)^T d| <= lbfgs_parameter_t::wolfe * |g(x)^T d|,
# where x is the current point, d is the current search direction,
# and a is the step length.
LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3


def _cubic_minimizer(
    u: float,
    fu: float,  # f(u)
    du: float,  # f'(u)
    v: float,
    fv: float,  # f(v)
    dv: float,  # f'(v)
) -> float:
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    p_val = abs(theta)
    q_val = abs(du)
    r_val = abs(dv)
    s = max(p_val, q_val, r_val)
    a = theta / s
    gamma = s * math.sqrt(a * a - (du / s) * (dv / s))
    if v < u:
        gamma = -gamma
    p_val = gamma - du + theta
    q_val = gamma - du + gamma + dv
    r_ratio = p_val / q_val
    return u + r_ratio * d


def _cubic_minimizer2(
    u: float,
    fu: float,  # f(u)
    du: float,  # f'(u)
    v: float,
    fv: float,  # f(v)
    dv: float,  # f'(v)
    xmin: float,
    xmax: float,
) -> float:
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    p_val = abs(theta)
    q_val = abs(du)
    r_val = abs(dv)
    s = max(p_val, q_val, r_val)
    a = theta / s
    gamma = s * math.sqrt(max(0, a * a - (du / s) * (dv / s)))
    if u < v:
        gamma = -gamma
    p_val = gamma - dv + theta
    q_val = gamma - dv + gamma + du
    r_ratio = p_val / q_val
    if r_ratio < 0.0 and gamma != 0.0:
        return v - r_ratio * d
    elif d > 0:
        return xmax
    else:
        return xmin


def _quad_minimizer(
    u: float,
    fu: float,  # f(u)
    du: float,  # f'(u)
    v: float,
    fv: float,  # f(v)
) -> float:
    a = v - u
    return u + du / ((fu - fv) / a + du) * (a / 2)


def _quad_minimizer2(
    u: float,
    du: float,  # f'(u)
    v: float,
    dv: float,  # f'(v)
) -> float:
    a = u - v
    return v + dv / (dv - du) * a


def _update_trial_interval(
    x: float,
    fx: float,
    dx: float,
    y: float,
    fy: float,
    dy: float,
    t: float,
    ft: float,
    dt: float,
    tmin: float,
    tmax: float,
    brackt: bool,
) -> Tuple[float, float, float, float, float, float, float, bool, RetCode]:
    """
    Returns updated values (x, fx, dx, y, fy, dy, t, brackt, status).
    In the original function the pointers are updated in-place.
    Error codes are returned as status (0 means normal termination).

    Jorge J. More and David J. Thuente. Line search algorithm with
    guaranteed sufficient decrease. ACM Transactions on Mathematical
    Software (TOMS), Vol 20, No 3, pp. 286-307, 1994.
    """
    # Pre-check if the trial value is out of interval, etc.
    if brackt:
        if t <= min(x, y) or max(x, y) <= t:
            # The trial value t is out of the interval.
            return x, fx, dx, y, fy, dy, t, brackt, RetCode.ERR_OUTOFINTERVAL
        if 0.0 <= dx * (t - x):
            # The function must decrease from x.
            return x, fx, dx, y, fy, dy, t, brackt, RetCode.ERR_INCREASEGRADIENT
        if tmax < tmin:
            # Incorrect tmin and tmax specified.
            return x, fx, dx, y, fy, dy, t, brackt, RetCode.ERR_INCORRECT_TMINMAX

    bound = 0
    dsign = dt * dx < 0.0
    newt = 0.0

    # Case 1: f(x) < f(t)
    if fx < ft:
        brackt = True
        bound = 1
        mc = _cubic_minimizer(x, fx, dx, t, ft, dt)
        mq = _quad_minimizer(x, fx, dx, t, ft)
        if abs(mc - x) < abs(mq - x):
            newt = mc
        else:
            newt = mc + 0.5 * (mq - mc)
    # Case 2: f(t) <= f(x) and derivatives have opposite sign.
    elif dsign:
        brackt = True
        bound = 0
        mc = _cubic_minimizer(x, fx, dx, t, ft, dt)
        mq = _quad_minimizer2(x, dx, t, dt)
        if abs(mc - t) > abs(mq - t):
            newt = mc
        else:
            newt = mq
    # Case 3: f(t) <= f(x) and |dt| < |dx|
    elif abs(dt) < abs(dx):
        bound = 1
        mc = _cubic_minimizer2(x, fx, dx, t, ft, dt, tmin, tmax)
        mq = _quad_minimizer2(x, dx, t, dt)
        if brackt:
            if abs(t - mc) < abs(t - mq):
                newt = mc
            else:
                newt = mq
        else:
            if abs(t - mc) > abs(t - mq):
                newt = mc
            else:
                newt = mq
    # Case 4: f(t) <= f(x) and |dt| is not smaller than |dx|.
    else:
        bound = 0
        if brackt:
            newt = _cubic_minimizer(t, ft, dt, y, fy, dy)
        elif x < t:
            newt = tmax
        else:
            newt = tmin

    # Update the interval of uncertainty.
    # - Case a: if f(x) < f(t),
    #     x <- x, y <- t.
    # - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
    #     x <- t, y <- y.
    # - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
    #     x <- t, y <- x.
    if fx < ft:
        # Case a
        y = t
        fy = ft
        dy = dt
    else:
        # Case c
        if dsign:
            y = x
            fy = fx
            dy = dx
        # Case b and c
        x = t
        fx = ft
        dx = dt

    # Clip the new trial value within [tmin, tmax].
    if newt > tmax:
        newt = tmax
    if newt < tmin:
        newt = tmin

    # Redefine newt if it is close to the upper bound of the interval.
    if brackt and bound:
        mq_val = x + 0.66 * (y - x)
        if x < y:
            if mq_val < newt:
                newt = mq_val
        else:
            if newt < mq_val:
                newt = mq_val

    # Set the new trial value.
    t = newt
    return x, fx, dx, y, fy, dy, t, brackt, RetCode.LINESEARCH_SUCCESS


def _owlqn_x1norm(x: npt.NDArray[np.float64], start: int, end: int) -> float:
    return float(np.sum(np.abs(x[start:end])))


def owlqn_pseudo_gradient(
    x: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    n: int,
    param: LBFGSParameter,
) -> npt.NDArray[np.float64]:
    pg = np.copy(g)
    for i in range(param.orthantwise_start, param.orthantwise_end):
        if x[i] < 0.0:
            # Differentiable.
            pg[i] = g[i] - param.orthantwise_c
        elif x[i] > 0.0:
            # Differentiable.
            pg[i] = g[i] + param.orthantwise_c
        else:
            if g[i] < -param.orthantwise_c:
                # Take the right partial derivative.
                pg[i] = g[i] + param.orthantwise_c
            elif param.orthantwise_c < g[i]:
                # Take the left partial derivative.
                pg[i] = g[i] - param.orthantwise_c
            else:
                pg[i] = 0.0
    return pg


def _owlqn_project(
    d: npt.NDArray[np.float64], sign: npt.NDArray[np.float64], start: int, end: int
) -> None:
    d[start:end][d[start:end] * sign[start:end] <= 0] = 0


def line_search_backtracking(
    n: int,
    x: npt.NDArray[np.float64],
    f: float,
    g: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    stp: float,
    xp: npt.NDArray[np.float64],
    gp: npt.NDArray[np.float64],
    wp: npt.NDArray[np.float64],
    cd: CallbackData,
    param: LBFGSParameter,
) -> Tuple[RetCode, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = 0
    dec, inc = 0.5, 2.1

    if stp <= 0.0:
        return RetCode.ERR_INVALIDPARAMETERS, f, stp, x, g

    dginit = np.dot(g, s)
    if dginit > 0:
        return RetCode.ERR_INCREASEGRADIENT, f, stp, x, g

    finit = f
    dgtest = param.ftol * dginit

    while True:
        x[:] = xp[:]
        x += s * stp

        f, g = cd.instance.evaluate(x)
        count += 1

        if f > finit + stp * dgtest:
            width = dec
        else:
            # Sufficient decrease (Armijo condition) met
            if param.linesearch_kind == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO:
                return RetCode.LINESEARCH_SUCCESS, f, stp, x, g
            # Check Wolfe condition
            dg = np.dot(g, s)
            if dg < param.wolfe * dginit:
                width = inc
            else:
                if param.linesearch_kind == LBFGS_LINESEARCH_BACKTRACKING_WOLFE:
                    return RetCode.LINESEARCH_SUCCESS, f, stp, x, g
                if dg > -param.wolfe * dginit:
                    width = dec
                else:
                    # Strong Wolfe condition is met
                    return RetCode.LINESEARCH_SUCCESS, f, stp, x, g

        if stp < param.min_step:
            return RetCode.ERR_MINIMUMSTEP, f, stp, x, g
        if stp > param.max_step:
            return RetCode.ERR_MAXIMUMSTEP, f, stp, x, g
        if param.max_linesearch <= count:
            return RetCode.ERR_MAXIMUMLINESEARCH, f, stp, x, g

        stp *= width


def line_search_backtracking_owlqn(
    n: int,
    x: npt.NDArray[np.float64],
    f: float,
    g: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    stp: float,
    xp: npt.NDArray[np.float64],
    gp: npt.NDArray[np.float64],
    wp: npt.NDArray[np.float64],
    cd: CallbackData,
    param: LBFGSParameter,
) -> Tuple[RetCode, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = 0
    width = 0.5
    norm = 0.0
    finit = f

    if stp <= 0.0:
        return RetCode.ERR_INVALIDPARAMETERS, f, stp, x, g

    # Choose the orthant: if xp[i] is zero, use -gp[i], else xp[i]
    wp[:] = np.where(xp == 0.0, -gp, xp)

    while True:
        # x <- xp; then x = x + stp * s
        x[:] = xp[:]
        x += s * stp
        # Project x onto the orthant defined by wp. (Assumes owlqn_project is defined.)
        _owlqn_project(x, wp, param.orthantwise_start, param.orthantwise_end)
        # Evaluate f and gradient g.
        f, g = cd.instance.evaluate(x)
        # Compute L1 norm for OWL-QN and add penalty.
        norm = _owlqn_x1norm(x, param.orthantwise_start, param.orthantwise_end)
        f += norm * param.orthantwise_c
        count += 1

        # Compute dgtest as sum((x - xp) * gp)
        dgtest = np.sum((x - xp) * gp)

        if f <= finit + param.ftol * dgtest:
            return RetCode.LINESEARCH_SUCCESS, f, stp, x, g
        if stp < param.min_step:
            return RetCode.ERR_MINIMUMSTEP, f, stp, x, g
        if stp > param.max_step:
            return RetCode.ERR_MAXIMUMSTEP, f, stp, x, g
        if param.max_linesearch <= count:
            return RetCode.ERR_MAXIMUMLINESEARCH, f, stp, x, g

        stp *= width


def line_search_morethuente(
    n: int,
    x: npt.NDArray[np.float64],
    f: float,
    g: npt.NDArray[np.float64],
    s: npt.NDArray[np.float64],
    stp: float,
    xp: npt.NDArray[np.float64],
    gp: npt.NDArray[np.float64],
    wp: npt.NDArray[np.float64],
    cd: CallbackData,
    param: LBFGSParameter,
) -> Tuple[RetCode, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = 0
    brackt = False
    stage1 = True
    uinfo = 0

    if stp <= 0.0:
        return RetCode.ERR_INVALIDPARAMETERS, f, stp, x, g

    dginit = np.dot(g, s)
    if dginit > 0:
        return RetCode.ERR_INCREASEGRADIENT, f, stp, x, g

    finit = f
    dgtest = param.ftol * dginit
    width = param.max_step - param.min_step
    prev_width = 2.0 * width

    stx = 0.0
    fx = finit
    dgx = dginit
    sty = 0.0
    fy = finit
    dgy = dginit

    while True:
        # Set stmin and stmax based on bracketing.
        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stx
            stmax = stp + 4.0 * (stp - stx)

        # Clip stp within [param.min_step, param.max_step]
        if stp < param.min_step:
            stp = param.min_step
        if stp > param.max_step:
            stp = param.max_step

        # Unusual termination: if conditions are met, choose stx
        if (
            brackt
            and (
                stp <= stmin
                or stmax <= stp
                or param.max_linesearch <= count + 1
                or uinfo != 0
            )
        ) or (brackt and (stmax - stmin <= param.xtol * stmax)):
            stp = stx

        # x <- xp; then x = x + stp * s.
        x[:] = xp[:]
        x += s * stp
        # Evaluate function and compute directional derivative.
        f, g = cd.instance.evaluate(x)
        dg = np.dot(g, s)
        ftest1 = finit + stp * dgtest
        count += 1

        if brackt and (stp <= stmin or stmax <= stp or uinfo != 0):
            return RetCode.ERR_ROUNDING_ERROR, f, stp, x, g
        if stp == param.max_step and f <= ftest1 and dg <= dgtest:
            return RetCode.ERR_MAXIMUMSTEP, f, stp, x, g
        if stp == param.min_step and (ftest1 < f or dgtest <= dg):
            return RetCode.ERR_MINIMUMSTEP, f, stp, x, g
        if brackt and (stmax - stmin) <= param.xtol * stmax:
            return RetCode.ERR_OUTOFINTERVAL, f, stp, x, g
        if param.max_linesearch <= count:
            return RetCode.ERR_MAXIMUMLINESEARCH, f, stp, x, g
        if f <= ftest1 and abs(dg) <= param.gtol * (-dginit):
            return RetCode.LINESEARCH_SUCCESS, f, stp, x, g

        if stage1 and f <= ftest1 and min(param.ftol, param.gtol) * dginit <= dg:
            stage1 = False

        # Use modified function if in stage1.
        if stage1 and ftest1 < f and f <= fx:
            fm = f - stp * dgtest
            fxm = fx - stx * dgtest
            fym = fy - sty * dgtest
            dgm = dg - dgtest
            dgxm = dgx - dgtest
            dgym = dgy - dgtest
            # Update trial interval (using update_trial_interval below)
            # Note: In Python, we simulate pointer updates by receiving updated values.
            # x, fx, dx, y, fy, dy, t, brackt, 0
            (stx, fxm, dgxm, sty, fym, dgym, stp, brackt, uinfo) = (
                _update_trial_interval(
                    stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, stmin, stmax, brackt
                )
            )
            fx = fxm + stx * dgtest
            fy = fym + sty * dgtest
            dgx = dgxm + dgtest
            dgy = dgym + dgtest
        else:
            (stx, fx, dgx, sty, fy, dgy, stp, brackt, uinfo) = _update_trial_interval(
                stx, fx, dgx, sty, fy, dgy, stp, f, dg, stmin, stmax, brackt
            )

        if brackt:
            if 0.66 * prev_width <= abs(sty - stx):
                stp = stx + 0.5 * (sty - stx)
            prev_width = width
            width = abs(sty - stx)
