import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qnlab.util.doc_paths import doc_imgs_dir

OUTPUT_DIR = doc_imgs_dir("quasi_newton")

# Set seaborn style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")


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
    # Different y range settings for different lambda values
    if lambda_ == 100:
        xmin, xmax = -1.2, 1.2
        ymin, ymax = -1.2, 1.2
    else:
        xmin, xmax = -2.0, 2.0
        ymin, ymax = -3.0, 1.0

    # Create visualization with 2x1 layout
    # Fixed figsize for consistent appearance
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")

    x_range = np.linspace(xmin, xmax, 150)
    y_range = np.linspace(ymin, ymax, 150)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 0.5 * (X**2 + Y**2)  # quadratic function f(x) = 0.5 * ||x||^2

    # Color for contour lines
    contour_color = "#2C3E50"

    # Plot 1: Function contour with BFGS trajectory
    axes[0].contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    axes[0].contour(X, Y, Z, levels=20, colors=contour_color, linewidths=0.5, alpha=0.4)

    # Plot BFGS trajectory with improved styling
    axes[0].plot(
        x_bfgs[:, 0],
        x_bfgs[:, 1],
        "o-",
        color="#E74C3C",
        markersize=6,
        linewidth=2.5,
        label="BFGS",
        alpha=0.85,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    axes[0].plot(
        x_bfgs[0, 0],
        x_bfgs[0, 1],
        "o",
        color="#27AE60",
        markersize=12,
        label="Start",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    axes[0].plot(
        0,
        0,
        "*",
        color="#F39C12",
        markersize=20,
        label="Minimum",
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )

    axes[0].set_title(
        f"BFGS Trajectory (λ={lambda_})", fontsize=18, fontweight="bold", pad=20
    )
    axes[0].set_xlabel("$x_1$", fontsize=16, fontweight="bold")
    axes[0].set_ylabel("$x_2$", fontsize=16, fontweight="bold")
    axes[0].legend(
        fontsize=13, framealpha=0.95, loc="best", edgecolor="black", fancybox=True
    )
    axes[0].tick_params(axis="both", which="major", labelsize=12)
    axes[0].grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    axes[0].set_aspect("equal")

    # Plot 2: Function contour with DFP trajectory
    axes[1].contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    axes[1].contour(X, Y, Z, levels=20, colors=contour_color, linewidths=0.5, alpha=0.4)

    # Plot DFP trajectory with improved styling
    axes[1].plot(
        x_dfp[:, 0],
        x_dfp[:, 1],
        "o-",
        color="#3498DB",
        markersize=6,
        linewidth=2.5,
        label="DFP",
        alpha=0.85,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    axes[1].plot(
        x_dfp[0, 0],
        x_dfp[0, 1],
        "o",
        color="#27AE60",
        markersize=12,
        label="Start",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    axes[1].plot(
        0,
        0,
        "*",
        color="#F39C12",
        markersize=20,
        label="Minimum",
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )

    axes[1].set_title(
        f"DFP Trajectory (λ={lambda_})", fontsize=18, fontweight="bold", pad=20
    )
    axes[1].set_xlabel("$x_1$", fontsize=16, fontweight="bold")
    axes[1].set_ylabel("$x_2$", fontsize=16, fontweight="bold")
    axes[1].legend(
        fontsize=13, framealpha=0.95, loc="best", edgecolor="black", fancybox=True
    )
    axes[1].tick_params(axis="both", which="major", labelsize=12)
    axes[1].grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"bfgs_vs_dfp_{lambda_}.pdf", bbox_inches="tight", dpi=300)
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
