import os

import numpy as np
from scipy.optimize import line_search, minimize

from qnlab.experiment.vis import vis
from qnlab.problem.base import BaseProblem
from qnlab.problem.rosenbrock import RosenbrockProblem
from qnlab.util.callback import Callback

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def gradient_descent_with_line_search(
    prob: BaseProblem, callback: Callback, max_iter=1000, tol=1e-6
):
    """Gradient descent with scipy line search"""
    x = prob.x0.copy()
    fx = prob.f(x, count=False)
    g = prob.g(x, count=False)

    callback.start(prob, x)

    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

        # Direction: negative gradient
        d = -g

        alpha, fc, gc, fx_new, g_old, g_new = line_search(prob.f, prob.g, x, d, g, fx)
        x_new = x + alpha * d
        g_new = prob.g(x_new)

        x = x_new
        fx = fx_new
        g = g_new

        callback.callback(prob, x, fx, g)

    return x, fx


def newton_with_line_search(
    prob: BaseProblem, callback: Callback, max_iter=1000, tol=1e-6
):
    """Newton method with line search"""
    x = prob.x0.copy()
    fx = prob.f(x, count=False)
    g = prob.g(x, count=False)

    callback.start(prob, x)

    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

        H = prob._hessian(x)
        d = np.linalg.solve(H, -g)

        alpha, fc, gc, fx_new, g_old, g_new = line_search(prob.f, prob.g, x, d, g, fx)
        assert isinstance(alpha, float)
        x_new = x + alpha * d
        g_new = prob.g(x_new)

        x = x_new
        fx = fx_new
        g = g_new

        callback.callback(prob, x, fx, g)

    return x, fx


def scipy_lbfgsb_with_callback(
    prob: BaseProblem, callback: Callback, max_iter=1000, tol=1e-6
):
    """L-BFGS-B using scipy.optimize.minimize with callback"""
    x0 = prob.x0.copy()

    callback.start(prob, x0)

    def scipy_callback(x):
        fx = prob.f(x, count=False)
        g = prob.g(x, count=False)
        callback.callback(prob, x, fx, g)

    result = minimize(
        prob.f,
        x0,
        method="L-BFGS-B",
        jac=prob.g,
        callback=scipy_callback,
        options={"maxiter": max_iter, "gtol": tol, "ftol": 0},
    )

    return result.x, result.fun


def main():
    """Compare three optimization methods on Rosenbrock problem"""
    # Problem setup
    n = 2  # Use 2D for visualization
    prob = RosenbrockProblem(n)

    # Callbacks for each method
    callback_gd = Callback(save_xs=True)
    callback_newton = Callback(save_xs=True)
    callback_lbfgsb = Callback(save_xs=True)

    print("Solving Rosenbrock problem with three methods...")
    print(f"Initial point: {prob.x0}")
    print(f"Optimal point: {prob.x_opt}")
    print(f"Initial function value: {prob.f(prob.x0):.6f}")
    print()

    print("1. Gradient Descent with Line Search")
    gradient_descent_with_line_search(prob, callback_gd)

    print("2. Newton Method with Line Search")
    newton_with_line_search(prob, callback_newton)

    print("3. L-BFGS-B (SciPy)")
    scipy_lbfgsb_with_callback(prob, callback_lbfgsb)

    callbacks = [callback_gd, callback_newton, callback_lbfgsb]
    labels = ["Gradient Descent", "Newton Method", "Quasi Newton"]

    vis(
        prob,
        callbacks,
        labels,
        "Rosenbrock",
        pdf_path="newton_vs_qs_vs_gd",
        max_length=100,
        use_tex=True,
        one_figure=True,
        x_axis="iterations",
    )


if __name__ == "__main__":
    main()
