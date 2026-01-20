import math
import os

import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def quadratic_f(x):
    return 0.5 * float(np.dot(x, x))


def grad_f(x):
    return x.copy()


def run_method(method, lambda_1, eps=1e-4, max_iter=20000, verbose=False):
    # Initial B1: eigenvalues 1 and lambda_1, choose eigenvectors aligned with axes
    B = np.diag([1.0, float(lambda_1)])

    # initial x1: angle theta satisfying tan^2(theta) = lambda_1  (Powell eq (4.2))
    theta = math.atan(math.sqrt(lambda_1))
    x = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    x0_norm = np.linalg.norm(x)

    f_vals = [quadratic_f(x)]
    x_trajectory = [x.copy()]

    # iterate
    for k in range(max_iter):
        g = grad_f(x)
        # search direction: -B^{-1} g  (solve linear system)
        p = -np.linalg.solve(B, g)
        s = p  # since step length = 1
        x_next = x + s
        y = grad_f(x_next) - grad_f(x)  # for quadratic, y = s

        sTy = float(np.dot(s, y))
        assert abs(sTy) > 0

        Bs = B.dot(s)
        sTBs = float(np.dot(s, Bs))

        if method == "BFGS":
            B = B + np.outer(y, y) / sTy - np.outer(Bs, Bs) / sTBs
        elif method == "DFP":
            # Use dual BFGS update formula for DFP
            IRhoSY = np.eye(2) - np.outer(s, y) / sTy
            B = IRhoSY.T @ B @ IRhoSY + np.outer(y, y) / sTy
        else:
            raise ValueError(f"Unknown method: {method}")

        x = x_next
        f_vals.append(quadratic_f(x))
        x_trajectory.append(x.copy())

        if np.linalg.norm(x) < eps * x0_norm:
            return k + 1, np.array(f_vals), np.array(x_trajectory)

    return None, np.array(f_vals), np.array(x_trajectory)


def plot_trajectories(lambda_, x_bfgs, x_dfp):
    # Create grid for function visualization
    default_min, default_max = -1.5, 1.5

    xmin = min(x_bfgs[:, 0].min(), x_dfp[:, 0].min(), default_min)
    xmax = max(x_bfgs[:, 0].max(), x_dfp[:, 0].max(), default_max)
    ymin = min(x_bfgs[:, 1].min(), x_dfp[:, 1].min(), default_min)
    ymax = max(x_bfgs[:, 1].max(), x_dfp[:, 1].max(), default_max)

    x_pad = 0.1 * (xmax - xmin)
    y_pad = 0.1 * (ymax - ymin)
    if xmin < default_min:
        xmin -= x_pad
    if xmax > default_max:
        xmax += x_pad
    if ymin < default_min:
        ymin -= y_pad
    if ymax > default_max:
        ymax += y_pad

    # Create visualization with 2x1 layout (removing axes[1,0] and axes[1,1])
    fig, axes = plt.subplots(
        1, 2, figsize=(1 + 2 * 6 * (xmax - xmin) / (ymax - ymin), 6)
    )

    x_range = np.linspace(xmin, xmax, 100)
    y_range = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 0.5 * (X**2 + Y**2)  # quadratic function f(x) = 0.5 * ||x||^2

    # Plot 1: Function contour with BFGS trajectory
    axes[0].contour(X, Y, Z, levels=20, alpha=0.9)
    axes[0].plot(
        x_bfgs[:, 0],
        x_bfgs[:, 1],
        "ro-",
        markersize=5,
        linewidth=2.5,
        label="BFGS",
        alpha=0.8,
    )
    axes[0].plot(x_bfgs[0, 0], x_bfgs[0, 1], "go", markersize=10, label="Start")
    axes[0].plot(0, 0, "k*", markersize=12, label="Minimum")
    axes[0].set_title(f"BFGS Trajectory (lambda={lambda_})", fontsize=14)
    axes[0].set_xlabel("x₁", fontsize=12)
    axes[0].set_ylabel("x₂", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    # Plot 2: Function contour with DFP trajectory
    axes[1].contour(X, Y, Z, levels=20, alpha=0.9)
    axes[1].plot(
        x_dfp[:, 0],
        x_dfp[:, 1],
        "bo-",
        markersize=5,
        linewidth=2.5,
        label="DFP",
        alpha=0.8,
    )
    axes[1].plot(x_dfp[0, 0], x_dfp[0, 1], "go", markersize=10, label="Start")
    axes[1].plot(0, 0, "k*", markersize=12, label="Minimum")
    axes[1].set_title(f"DFP Trajectory (lambda={lambda_})", fontsize=14)
    axes[1].set_xlabel("x₁", fontsize=12)
    axes[1].set_ylabel("x₂", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"bfgs_vs_dfp_{lambda_}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"bfgs_vs_dfp_{lambda_}.pdf", bbox_inches="tight")
    plt.close()


def compare_convergence(lambda_values, eps=1e-4, max_iter=20000):
    print("\nSummary Results:")

    # Store results for table generation
    results = []

    for lambda_ in lambda_values:
        it_bfgs, f_bfgs, x_bfgs = run_method(
            "BFGS", lambda_, eps=eps, max_iter=max_iter
        )
        it_dfp, f_dfp, x_dfp = run_method("DFP", lambda_, eps=eps, max_iter=max_iter)

        bfgs_str = str(it_bfgs) if it_bfgs is not None else ">=max"
        dfp_str = str(it_dfp) if it_dfp is not None else ">=max"

        if lambda_ in [0.1, 100]:
            plot_trajectories(lambda_, x_bfgs, x_dfp)

        # Store results for table
        results.append(
            {
                "lambda": lambda_,
                "BFGS": bfgs_str,
                "DFP": dfp_str,
                "BFGS_raw": it_bfgs,
                "DFP_raw": it_dfp,
            }
        )

    # Generate Markdown table
    print("\n" + "=" * 50)
    print("Markdown Table:")
    print("=" * 50)
    print("| $\\lambda_1$ | BFGS | DFP |")
    print("|---|---|---|")
    for result in results:
        print(f"| {result['lambda']} | {result['BFGS']} | {result['DFP']} |")

    # Generate LaTeX table
    print("\n" + "=" * 50)
    print("LaTeX Table:")
    print("=" * 50)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|}")
    print("\\hline")
    print("$\\lambda_1$ & BFGS & DFP \\\\")
    print("\\hline")
    for result in results:
        a_latex = f"{result['lambda']}"
        print(f"{a_latex} & {result['BFGS']} & {result['DFP']} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Convergence comparison between BFGS and DFP methods}")
    print("\\label{tab:bfgs_dfp_comparison}")
    print("\\end{table}")


def main():
    """Main execution function"""
    # Parameters
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    eps = 1e-4
    max_iter = 20000

    print("Running Powell-like simulation: BFGS vs DFP")
    compare_convergence(lambda_values, eps, max_iter)


if __name__ == "__main__":
    main()
