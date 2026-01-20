import os

import matplotlib.pyplot as plt
import numpy as np


def make_random_convex_like_function(seed):
    """Create a random 1D convex-like function with positive second derivative at origin."""
    rng = np.random.default_rng(seed)

    for attempt in range(100):
        # Random parameters for 1D function: f(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
        c0 = rng.standard_normal()
        c1 = rng.standard_normal()
        c2 = rng.standard_normal()
        c3 = rng.standard_normal()
        c4 = rng.standard_normal()

        # Second derivative at x=0 is 2*c2
        if 2 * c2 > 1e-6:  # Ensure positive curvature at origin

            def f(x):
                x = np.asarray(x)[0]
                return np.float64(c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4)

            def gf(x):
                x = np.asarray(x)[0]
                return np.array([c1 + 2 * c2 * x + 3 * c3 * x**2 + 4 * c4 * x**3])

            def hf(x):
                x = np.asarray(x)[0]
                return np.float64(2 * c2 + 6 * c3 * x + 12 * c4 * x**2)

            return f, gf, hf, (c0, c1, c2, c3, c4)

    raise RuntimeError("Failed to generate function with positive curvature.")


def compute_sigma_zhang(s, g, gp, fx, fp):
    ss = s * s
    diff_f = fp - fx
    s_ggp = s * (g + gp)
    sigma = (6 * diff_f + 3 * s_ggp) / ss
    return sigma


def compute_sigma_wei(s, g, gp, fx, fp):
    ss = s * s
    diff_f = fp - fx
    s_ggp = s * (g + gp)
    sigma = (2.0 * diff_f + s_ggp) / ss
    return sigma


def compute_sigma_yuan(s, g, gp, fx, fp):
    ss = s * s
    diff_f = fp - fx
    s_ggp = s * (g + gp)
    sigma = max(0.0, 2.0 * diff_f + s_ggp) / ss
    return sigma


def compute_sigma_hamaguchi(s, g, gp, fx, fp):
    if fp < fx:
        return np.float64(0.0)
    ss = s * s
    diff_f = fp - fx
    s_ggp = np.dot(s, g + gp)
    # Zhang et al. 1999
    sigma_Z = (6 * diff_f + 3 * s_ggp) / ss
    # coeff=2 ver: Wei et al. 2006
    # taking max of coeff=2 ver: Yuan et al. 2017
    sigma_Y = max(0.0, 2.0 * diff_f + s_ggp) / ss
    sigma = (0 + sigma_Y + sigma_Z) / 3
    return sigma


def compute_modified_secant_result(s, y, g, fx, fxp, gp, sigma_func):
    """Compute modified secant y and its direction.

    Returns:
        dict with keys: y_mod, ys_mod, yy_mod, sigma, d, success
    """
    sigma = sigma_func(s, g, gp, fx, fxp)
    y_mod = y + sigma * s
    ys_mod = y_mod * s
    yy_mod = y_mod * y_mod
    d = -g * (ys_mod / yy_mod)

    return {
        "success": True,
        "y_mod": y_mod,
        "ys_mod": ys_mod,
        "yy_mod": yy_mod,
        "sigma": sigma,
        "d": d,
    }


def visualize_comparison(seed):
    sigma_functions = {
        "raw": lambda s, g, gp, fx, fp: 0.0,
        "Zhang": compute_sigma_zhang,
        "Wei": compute_sigma_wei,
        "Yuan": compute_sigma_yuan,
        "Hamaguchi": compute_sigma_hamaguchi,
    }

    np.random.seed(seed)

    try:
        f, gf, hf, params_tuple = make_random_convex_like_function(seed)
    except RuntimeError as e:
        print(f"Seed {seed}: {e}")
        return None

    # Sample points and their function/gradient values (1D)
    x = np.random.randn()
    xp = np.random.randn()
    fx = f(np.array([x]))
    fxp = f(np.array([xp]))
    g = gf(np.array([x]))[0]
    gp = gf(np.array([xp]))[0]

    # Compute s, y and their products
    s = x - xp
    y = g - gp
    ss = s * s
    # ys = y * s
    yy = y * y

    if not (np.isfinite(ss) and np.isfinite(yy) and ss > 0):  # and ys > 0):
        print(f"Seed {seed}: Invalid s or y")
        return None

    methods = {}

    for name, sigma_func in sigma_functions.items():
        result = compute_modified_secant_result(s, y, g, fx, fxp, gp, sigma_func)
        if not result["success"]:
            print(f"Seed {seed}: {name} failed")
            continue

        x_next = x + result["d"]
        f_next = f(np.array([x_next]))

        if not np.isfinite(f_next):
            print(f"Seed {seed}: {name} non-finite")
            continue

        methods[name] = {
            "f_next": f_next,
            "x_next": x_next,
            "y": result["y_mod"],
            "sigma": result["sigma"],
        }

    if len(methods) <= 1:
        print(f"Seed {seed}: No valid modified methods")
        return None

    # Find best method
    hessian_at_x = hf(np.array([x]))

    if hessian_at_x < 0:
        # Rank by |y/s| (smaller is better)
        method_scores = {name: abs(data["y"] / s) for name, data in methods.items()}
    else:
        # Rank by f_next (smaller is better)
        method_scores = {name: data["f_next"] for name, data in methods.items()}

    # Compute rankings for all methods
    sorted_methods_list = sorted(method_scores.items(), key=lambda x: x[1])

    # Assign ranks with tie handling (1st, 1st, 3rd style)
    method_ranks = {}
    current_rank = 1
    prev_score = None
    for i, (method, score) in enumerate(sorted_methods_list):
        if prev_score is not None and abs(score - prev_score) > 1e-8:
            current_rank = i + 1
        method_ranks[method] = current_rank
        prev_score = score

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the true function
    xmin = -2
    xmax = +2
    xVals = np.linspace(xmin, xmax, 1000)
    yVals = np.array([f(np.array([xi])) for xi in xVals])
    ax.plot(xVals, yVals, "-", color="black", linewidth=2, label="$f(x)$", zorder=1)

    # Plot current points
    ax.scatter(
        [xp],
        [fxp],
        color="black",
        s=100,
        marker="o",
        label="$x_{k-1}$",
        edgecolors="white",
        linewidths=1,
    )
    ax.scatter(
        [x],
        [fx],
        color="black",
        s=100,
        marker="D",
        label="$x_k$",
        edgecolors="white",
        linewidths=1,
    )

    # Plot quadratic models and next points
    for name, data in methods.items():
        y_model = fx + g * (xVals - x) + 0.5 * (data["y"] / s) * (xVals - x) ** 2
        # offset dash phase to avoid exact overlap between models
        idx = list(methods.keys()).index(name)
        dash_offset = idx * 2.0  # shift phase per method (adjust as needed)
        (line,) = ax.plot(
            xVals,
            y_model,
            linestyle=(dash_offset, (6, 4)),
            label=f"{name} Model",
        )
        if xmin <= data["x_next"] <= xmax:
            ax.scatter([data["x_next"]], [data["f_next"]], alpha=0.7)

    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim((xmin, xmax))
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$f(x)$", fontsize=12)
    ax.set_title(f"Comparison: Modified Secant Methods (Seed {seed})", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"compare_modified_vis{seed}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "seed": seed,
        "methods": {
            name: {"f_next": m["f_next"], "sigma": m["sigma"]}
            for name, m in methods.items()
        },
        "method_ranks": method_ranks,
        "is_negative_curvature": hessian_at_x < 0,
    }


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    results = []
    method_rankings: dict[str, dict[int, int]] = {}  # {method: {rank: count}}
    method_rankings2: dict[
        str, dict[int, int]
    ] = {}  # {method: {rank: count}} for negative curvature cases

    num_cases = 30
    print(f"Generating {num_cases} comparison visualizations...")
    for seed in range(num_cases):
        png_title = f"compare_modified_vis{seed}.png"
        if os.path.exists(png_title):
            os.remove(png_title)
        result = visualize_comparison(seed)
        if result is not None:
            results.append(result)
            method_ranks = result["method_ranks"]
            is_negative_curvature = result["is_negative_curvature"]

            # Update ranking counts
            target_rankings = (
                method_rankings2 if is_negative_curvature else method_rankings
            )
            for method, rank in method_ranks.items():
                if method not in target_rankings:
                    target_rankings[method] = {}
                target_rankings[method][rank] = target_rankings[method].get(rank, 0) + 1

            methods_str = ", ".join(
                f"{name}={data['f_next']:.4f}"
                for name, data in result["methods"].items()
            )
            print(f"Seed {seed}: {methods_str}")

    print("\n=== Summary ===")
    print(f"Total cases: {len(results)}")

    # Prepare and display/save tables with matplotlib instead of printing plain text
    def make_table_data(rankings):
        # rankings: {method: {rank: count}}
        methods = sorted(rankings.keys(), key=lambda m: -rankings[m].get(1, 0))
        if not methods:
            return ["No data"], [["-"]], ["No methods"]
        max_rank = max(
            (r for ranks in rankings.values() for r in ranks.keys()), default=0
        )
        col_labels = (
            ["Method"] + [f"{r}位" for r in range(1, max_rank + 1)] + ["Total", "1位率"]
        )
        rows = []
        for m in methods:
            ranks = rankings[m]
            counts = [ranks.get(r, 0) for r in range(1, max_rank + 1)]
            total = sum(ranks.values())
            first = ranks.get(1, 0)
            pct = f"{100 * first / total:.1f}%" if total > 0 else "0.0%"
            rows.append([m] + counts + [total, pct])
        return col_labels, rows, methods

    col_labels_pos, rows_pos, methods_pos = make_table_data(method_rankings)
    col_labels_neg, rows_neg, methods_neg = make_table_data(method_rankings2)

    # Determine figure size based on number of rows
    nrows = max(len(rows_pos), len(rows_neg), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(2.5, 0.5 * nrows + 1.5)))

    # Positive/zero curvature table
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Method rankings (positive/zero curvature cases)")
    if rows_pos and rows_pos != [["-"]]:
        table_pos = ax.table(
            cellText=rows_pos, colLabels=col_labels_pos, loc="center", cellLoc="center"
        )
        table_pos.auto_set_font_size(False)
        table_pos.set_fontsize(10)
        table_pos.scale(1, 1.2)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")

    # Negative curvature table
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Method rankings (negative curvature cases)")
    if rows_neg and rows_neg != [["-"]]:
        table_neg = ax.table(
            cellText=rows_neg, colLabels=col_labels_neg, loc="center", cellLoc="center"
        )
        table_neg.auto_set_font_size(False)
        table_neg.set_fontsize(10)
        table_neg.scale(1, 1.2)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    out_png = "compare_modified_table.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Also print a short textual summary
    print("\n=== Table saved to", out_png, "===")
    print(f"Total cases: {len(results)}")
