from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import line_search

from qnlab.parameter import NTRQNParameter
from qnlab.problem.base import BaseProblem
from qnlab.solver.qn_ntrqn import line_search_relaxed_armijo


class ExponentialProblem(BaseProblem):
    """Exponential Problem"""

    def __init__(self, scale1=1.0, scale2=0.0):
        x0 = np.array([0.0], dtype=np.float64)
        assert x0.ndim == 1
        self.scale = scale1
        self.scale2 = scale2
        self.x_hists = []
        super().__init__(f"Exponential_{scale1}_{scale2}", 1, x0)

    def _f(self, x: np.ndarray):
        assert x.ndim == 1
        self.x_hists.append(x.copy())
        return np.exp(self.scale * (x[0] + 1)) + self.scale2 * x[0] ** 2

    def _g(self, x: np.ndarray):
        """Gradient of the function."""
        g = self.scale * np.exp(self.scale * (x[0] + 1)) + 2 * self.scale2 * x[0]
        return np.array([g])


class ConvexEvenPolynomialProblem(BaseProblem):
    def __init__(self, degree: int = 4):
        if degree not in (2, 4, 6):
            raise ValueError("degree must be one of {2, 4, 6}.")
        self.degree = degree
        self.x_hists: list[np.ndarray] = []
        x0 = np.array([0.0], dtype=np.float64)
        super().__init__(f"ConvexEvenPolynomial_{degree}", 1, x0)

    def _f(self, x: npt.NDArray[np.float64]) -> np.float64:
        x0 = np.float64(x[0])
        if self.degree == 2:
            val = 0.5 * x0**2 - x0
        elif self.degree == 4:
            val = (1.0 / 4.0) * x0**4 + (1.0 / 2.0) * x0**2 - x0
        else:
            val = (1.0 / 6.0) * x0**6 + (1.0 / 4.0) * x0**4 + (1.0 / 2.0) * x0**2 - x0
        val += 5.0  # Shift to make f(x) > 0
        self.x_hists.append(x.copy())
        return np.float64(val)

    def _g(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x0 = np.float64(x[0])
        if self.degree == 2:
            grad = x0 - 1.0
        elif self.degree == 4:
            grad = x0**3 + x0 - 1.0
        else:
            grad = x0**5 + x0**3 + x0 - 1.0
        return np.array([grad], dtype=np.float64)


def run_ls(
    method_name: str,
    prob: Union[ExponentialProblem, ConvexEvenPolynomialProblem],
    x,
    c1: np.float64,
    c2: np.float64,
):
    prob.reset()
    prob.x_hists = [x]

    if method_name == "qn_ntrqn":
        fx = prob.f(x)
        g = prob.g(x)
        d = -g / np.linalg.norm(g)
        param = NTRQNParameter(n=1, options={"armijo": c1})
        _res, *_ = line_search_relaxed_armijo(
            x,
            fx,
            g,
            d,
            prob,
            param,
            prob.get_machine_eps(),
            fx,
            verbose=True,
            is_offo_mode=False,
            rejection_counter=0,
        )
    elif method_name == "scipy":
        fx = prob.f(x, count=False)
        g = prob.g(x, count=False)
        d = -g / np.linalg.norm(g)
        _res = line_search(prob.f, prob.g, xk=x, pk=d, gfk=g, old_fval=fx, c1=c1, c2=c2)
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    xs = [
        np.float64(xi[0]) if method_name == "qn_ntrqn" else float(xi[0])
        for xi in prob.x_hists
    ]
    ys = [prob.f(np.array([xi]), count=False) for xi in xs]
    gnorms = [abs(prob.g(np.array([xi]), count=False)[0]) for xi in xs]

    return xs, ys, gnorms


def plot_method_data(axes, xs, ys, gnorms, name):
    ax_f, ax_f_log, ax_g = axes

    color = "C0" if name == "ntrqn" else "C1"
    marker = "o" if name == "ntrqn" else "s"
    label = "qn_ntrqn" if name == "ntrqn" else "scipy (c2=0.01)"
    linestyle = "-" if name == "ntrqn" else "--"

    # Plot on function value plot (x vs f(x))
    ax_f.plot(
        xs,
        ys,
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=label,
        linewidth=2,
        alpha=0.7,
    )
    for i in range(len(xs) - 1):
        ax_f.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.5,
                "linestyle": linestyle,
            },
            alpha=0.7,
        )

    # Plot on function value log plot (evaluation step vs f(x))
    ax_f_log.plot(
        range(len(ys)),
        ys,
        marker=marker,
        color=color,
        label=label,
        linestyle=linestyle,
        linewidth=2,
        alpha=0.7,
    )

    # Plot on gradient norm log plot (evaluation step vs |g|)
    ax_g.plot(
        range(len(gnorms)),
        gnorms,
        marker=marker,
        color=color,
        label=label,
        linestyle=linestyle,
        linewidth=2,
        alpha=0.7,
    )


def visualize_opt_results(
    prob: Union[ExponentialProblem, ConvexEvenPolynomialProblem],
    c1: np.float64,
    c2: np.float64,
    output_dir: str,
):
    """Visualize line search for e^{100x} problem."""
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax_f, ax_f_log, ax_g = axes

    # Run qn_ntrqn line search
    xs_qn, y_path_qn, gnorms_qn = run_ls("qn_ntrqn", prob, prob.x0, c1, c2)
    plot_method_data((ax_f, ax_f_log, ax_g), xs_qn, y_path_qn, gnorms_qn, "ntrqn")

    # Run scipy line search
    xs_scipy, y_path_scipy, gnorms_scipy = run_ls("scipy", prob, prob.x0, c1, c2)
    plot_method_data(
        (ax_f, ax_f_log, ax_g), xs_scipy, y_path_scipy, gnorms_scipy, "scipy"
    )

    # Plot function value and search paths
    x_min = min(min(xs_qn), min(xs_scipy))
    x_max = max(max(xs_qn), max(xs_scipy))
    span = max(1e-3, x_max - x_min)
    margin = max(0.01, 0.1 * span)
    x_plot = np.linspace(x_min - margin, x_max + margin, 400)
    y_plot = [prob.f(np.array([xi])) for xi in x_plot]
    ax_f.plot(x_plot, y_plot, color="black", alpha=0.5, linewidth=2, label="f(x)")

    # Plot Armijo condition (c1 = 0.9)
    fx = prob.f(prob.x0, count=False)
    g = prob.g(prob.x0, count=False)
    fx0 = fx
    y_armijo = fx0 + c1 * (x_plot - prob.x0[0]) * g[0]
    ax_f.plot(
        x_plot,
        y_armijo,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Armijo (c1=0.9)",
        alpha=0.7,
    )

    # Configure ax_f
    ax_f.set_title(f"{prob.name} - Function value and search paths")
    ax_f.set_xlabel("x")
    ax_f.set_ylabel("f(x)")
    ax_f.legend()
    ax_f.grid(True, linestyle="--", alpha=0.4)

    # Configure ax_f_log
    ax_f_log.set_title(f"{prob.name} - Function value history")
    ax_f_log.set_xlabel("evaluation step")
    ax_f_log.set_ylabel("f(x)")
    ax_f_log.set_yscale("log")
    ax_f_log.legend()
    ax_f_log.grid(True, linestyle="--", alpha=0.4)

    # Configure ax_g
    ax_g.set_title(f"{prob.name} - Gradient norm history")
    ax_g.set_xlabel("evaluation step")
    ax_g.set_ylabel(r"\|g\|")
    ax_g.set_yscale("log")
    ax_g.legend()
    ax_g.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{prob.name}_c1_{c1}_c2_{c2}.pdf"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Saved visualization to {filepath}")
    else:
        plt.show()

    plt.close(fig)


def test_linesearch_qn_vs_scipy():
    """Test that qn_ntrqn line search results match scipy within tolerance."""
    tolerance = 1e-10

    test_cases = [
        (0.9, 0.99),
        (0.5, 0.9),
        (0.0001, 0.9),
    ]

    for c1, c2 in test_cases:
        # Test ExponentialProblem with different scales
        for scale2 in [0.0, 10.0, 1000.0]:
            prob = ExponentialProblem(scale1=1.0, scale2=scale2)
            xs_qn, _, _ = run_ls(
                "qn_ntrqn", prob, prob.x0, np.float64(c1), np.float64(c2)
            )

            prob_scipy = ExponentialProblem(scale1=1.0, scale2=scale2)
            xs_scipy, _, _ = run_ls(
                "scipy", prob_scipy, prob_scipy.x0, np.float64(c1), np.float64(c2)
            )

            # Truncate to the shorter length
            min_len = min(len(xs_qn), len(xs_scipy))
            xs_qn_truncated = xs_qn[:min_len]
            xs_scipy_truncated = xs_scipy[:min_len]

            # Check that all values are within tolerance
            for i, (qn_val, scipy_val) in enumerate(
                zip(xs_qn_truncated, xs_scipy_truncated)
            ):
                error = abs(qn_val - scipy_val)
                assert error < tolerance, (
                    f"ExponentialProblem(scale2={scale2}) at c1={c1}, c2={c2}: "
                    f"x[{i}] error {error} exceeds tolerance {tolerance}"
                )

        # Test ConvexEvenPolynomialProblem with different degrees
        for degree in [2, 4, 6]:
            prob = ConvexEvenPolynomialProblem(degree=degree)
            xs_qn, _, _ = run_ls(
                "qn_ntrqn", prob, prob.x0, np.float64(c1), np.float64(c2)
            )

            prob_scipy = ConvexEvenPolynomialProblem(degree=degree)
            xs_scipy, _, _ = run_ls(
                "scipy", prob_scipy, prob_scipy.x0, np.float64(c1), np.float64(c2)
            )

            # Truncate to the shorter length
            min_len = min(len(xs_qn), len(xs_scipy))
            xs_qn_truncated = xs_qn[:min_len]
            xs_scipy_truncated = xs_scipy[:min_len]

            # Check that all values are within tolerance
            for i, (qn_val, scipy_val) in enumerate(
                zip(xs_qn_truncated, xs_scipy_truncated)
            ):
                error = abs(qn_val - scipy_val)
                assert error < tolerance, (
                    f"ConvexEvenPolynomialProblem(degree={degree}) at c1={c1}, c2={c2}: "
                    f"x[{i}] error {error} exceeds tolerance {tolerance}"
                )

    print("Line search test passed successfully.")


if __name__ == "__main__":
    test_linesearch_qn_vs_scipy()

    yn = input("Run visualization? (y/n): ")
    if yn.lower() != "y":
        exit()

    # Visualize results (only when run directly)
    output_dir = Path(__file__).parent / "temp_imgs"

    for c1, c2 in [(0.9, 0.99), (0.5, 0.9), (0.0001, 0.9)]:
        for scale2 in [0.0, 10.0, 1000.0]:
            prob1 = ExponentialProblem(scale1=1.0, scale2=scale2)
            visualize_opt_results(
                prob1, np.float64(c1), np.float64(c2), str(output_dir)
            )
        for degree in [2, 4, 6]:
            prob2 = ConvexEvenPolynomialProblem(degree=degree)
            visualize_opt_results(
                prob2, np.float64(c1), np.float64(c2), str(output_dir)
            )
