import math
import os

import matplotlib.pyplot as plt
import numpy as np

# LaTeXを使用
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

os.chdir(os.path.dirname(__file__))


def hermite_cubic_function(xk, yk, ypk, xk1, yk1, ypk1):
    h = xk1 - xk
    if h == 0:
        raise ValueError("xk and xk1 must be different.")

    def p(x):
        t = (x - xk) / h
        H00 = 2 * t**3 - 3 * t**2 + 1
        H10 = t**3 - 2 * t**2 + t
        H01 = -2 * t**3 + 3 * t**2
        H11 = t**3 - t**2
        return H00 * yk + H10 * (h * ypk) + H01 * yk1 + H11 * (h * ypk1)

    return p


def _cubic_minimizer(
    u: np.float64,
    fu: np.float64,  # f(u)
    du: np.float64,  # f'(u)
    v: np.float64,
    fv: np.float64,  # f(v)
    dv: np.float64,  # f'(v)
) -> np.float64:
    """Computes the cubic minimizer for the line search.

    Returns:
        np.float64: Trial point for line search.
    """
    d = v - u
    theta = (fu - fv) * 3 / d + du + dv
    p_val = abs(theta)
    q_val = abs(du)
    r_val = abs(dv)
    s = np.max([p_val, q_val, r_val])
    a = theta / s
    gamma = s * math.sqrt(a * a - (du / s) * (dv / s))
    if v < u:
        gamma = -gamma
    p_val = gamma - du + theta
    q_val = gamma - du + gamma + dv
    r_ratio = p_val / q_val
    return u + r_ratio * d


def plot_modified_BFGS_demo():
    # パラメータ設定
    xk, yk, ypk = 0.0, 0.0, -1.0
    xk1 = 5.0
    ypk1 = 0.5

    # 3ケース: yk1が同じ, 小さい, 大きい
    cases = [
        (r"$f(x_{k}) = f(x_{k+1})$", 0.0),
        (r"$f(x_{k}) > f(x_{k+1})$", -2.0),
        (r"$f(x_{k}) < f(x_{k+1})$", 2.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    xs = np.linspace(xk - 0.5, xk1 + 0.5, 400)

    for ax, (title, yk1) in zip(axes, cases):
        p = hermite_cubic_function(xk, yk, ypk, xk1, yk1, ypk1)
        ys = p(xs)

        # 最小値を求める
        idx_min = np.argmin(ys)
        x_min, y_min = xs[idx_min], ys[idx_min]
        c_m = _cubic_minimizer(
            np.float64(xk),
            np.float64(yk),
            np.float64(ypk),
            np.float64(xk1),
            np.float64(yk1),
            np.float64(ypk1),
        )
        print(f"x_min:{x_min:.4f}, cubic_minimizer:{c_m:.4f}")

        # 曲線描画 (黒色)
        ax.plot(xs, ys, color="black", linewidth=4)
        ax.plot([xk, xk1], [yk, yk1], "o", color="black", markersize=15)
        ax.plot(x_min, y_min, "*", color="orange", markersize=20)

        # xk付近の範囲に限定した接線 (赤色)
        mask1 = (xs >= xk - 2) & (xs <= xk + 2)
        xs_t1 = xs[mask1]
        tangent_xk = ypk * (xs_t1 - xk) + yk
        ax.plot(xs_t1, tangent_xk, color="red", linewidth=3, linestyle="--", alpha=0.7)

        # xk1付近の範囲に限定した接線 (青色)
        mask2 = (xs >= xk1 - 2) & (xs <= xk1 + 2)
        xs_t2 = xs[mask2]
        tangent_xk1 = ypk1 * (xs_t2 - xk1) + yk1
        ax.plot(
            xs_t2, tangent_xk1, color="blue", linewidth=3, linestyle="--", alpha=0.7
        )

        # 軸設定
        ax.set_title(title, fontsize=30)

        # グリッド、枠、ticksを削除
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_ylabel(r"$f(x)$", fontsize=25)
    axes[1].set_xlabel(r"$x$", fontsize=25)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("cubic_interpolation.png", dpi=300)
    plt.savefig("cubic_interpolation.pdf")


if __name__ == "__main__":
    plot_modified_BFGS_demo()
