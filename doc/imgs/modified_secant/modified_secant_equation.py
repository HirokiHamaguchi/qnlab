from typing import Callable

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def compute_y1(x, fx, g, xp, fxp, gp) -> npt.NDArray[np.float64]:
    s = x - xp
    sigma = (6 * (fxp - fx) + 3 * np.dot(s, gp + g)) / np.sum(s**2)
    return sigma


def compute_y2(x, fx, g, xp, fxp, gp) -> npt.NDArray[np.float64]:
    s = x - xp
    sigma = (2.0 * (fxp - fx) + np.dot(s, g + gp)) / np.sum(s**2)
    return sigma


def compute_y3(x, fx, g, xp, fxp, gp) -> npt.NDArray[np.float64]:
    s = x - xp
    sigma = (6 * (fxp - fx) + 3 * np.dot(s, gp + g)) / np.sum(s**2)
    sigma1 = (4.0 * (fxp - fx) + np.dot(s, g + gp)) / np.sum(s**2)
    sigma2 = (1.0 * (fxp - fx) + np.dot(s, g + gp)) / np.sum(s**2)
    sigma = np.clip(sigma, min(sigma1, sigma2), max(sigma1, sigma2))
    return sigma


def trial(
    x_,
    xp_,
    f: Callable,
    f_prime: Callable,
    f_double_prime: Callable,
    xMin=-1.0,
    xMax=+1.0,
    kind: str = "",
):
    x = np.array([x_])
    fx = f(x)
    g = f_prime(x)
    h = f_double_prime(x)
    xp = np.array([xp_])
    fxp = f(xp)
    gp = f_prime(xp)
    _hp = f_double_prime(xp)

    xVals = np.linspace(xMin, xMax, 1000)
    yVals = f(xVals)

    bestX = -1.0
    bestY = f(bestX)

    yHESS = fx + g * (xVals - x) + 0.5 * h * (xVals - x) ** 2

    yBFGS = fx + g * (xVals - x) + 0.5 * (gp - g) / (xp - x) * (xVals - x) ** 2

    sigma1 = compute_y1(x, fx, g, xp, fxp, gp)
    g1 = gp - sigma1 * (x - xp)
    y1 = fx + g * (xVals - x) + 0.5 * (g1 - g) / (xp - x) * (xVals - x) ** 2

    sigma2 = compute_y2(x, fx, g, xp, fxp, gp)
    g2 = gp - sigma2 * (x - xp)
    y2 = fx + g * (xVals - x) + 0.5 * (g2 - g) / (xp - x) * (xVals - x) ** 2

    sigma3 = compute_y3(x, fx, g, xp, fxp, gp)
    g3 = gp - sigma3 * (x - xp)
    y3 = fx + g * (xVals - x) + 0.5 * (g3 - g) / (xp - x) * (xVals - x) ** 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_aspect("equal", adjustable="box")

    plt.plot(xVals, yVals, "-", color="tab:blue", label="$f(x)$")

    if "HESS" == kind:
        plt.plot(
            xVals, yHESS, color="black", label="Exact Quadratic Model", linewidth=3
        )
        # (bestX, bestY)を中心に赤色の円を描く
        circle = matplotlib.patches.Circle(
            (bestX, bestY), 0.05, color="red", fill=False, linestyle="--"
        )
        ax.add_artist(circle)
        plt.text(
            bestX - 0.2,
            bestY + 0.3,
            "Almost no gap",
            fontsize=30,
            color="red",
            zorder=5,
        )

    dx = 0.5
    fBest = np.nan

    if "EXPLAIN" == kind:
        plt.plot(
            [x[0] - dx, x[0] + dx],
            [fx - g * dx, fx + g * dx],
            "-.",
            color="darkgreen",
            zorder=-1,
        )
        plt.plot(
            [xp[0] - dx, xp[0] + dx],
            [fxp - gp * dx, fxp + gp * dx],
            "-.",
            color="darkgreen",
            zorder=-1,
        )

    if "BFGS" == kind:
        plt.plot(xVals, yBFGS, color="black", label="Quadratic Model", linewidth=3)
        fBFGS = fx + g * (xp - x) + 0.5 * (g - gp) / (x - xp) * (x - xp) ** 2
        plt.plot(
            [xp - dx, xp + dx],
            [fxp - gp * dx, fxp + gp * dx],
            "-.",
            color="darkgreen",
            zorder=-1,
        )
        plt.plot(
            [xp - dx, xp + dx],
            [fBFGS - gp * dx, fBFGS + gp * dx],
            "-.",
            color="darkgreen",
            zorder=-1,
        )
        fBest = (fx + g * (bestX - x) + 0.5 * (gp - g) / (xp - x) * (x - bestX) ** 2)[0]

    if "1_cubic" == kind or "1_quadratic" == kind:
        c = sigma1 / (3 * (x - xp) * np.abs(x - xp))
        cubic = (
            fx
            + g * (xVals - x)
            + 0.5 * (g - gp + sigma1 * (x - xp)) / (x - xp) * (xVals - x) ** 2
            + c * np.abs(xVals - x) ** 3
        )
        plt.plot(
            xVals,
            cubic,
            ":" if "1_quadratic" == kind else "--",
            color="black",
            label="Cubic Model",
            linewidth=3,
        )

        if "1_quadratic" == kind:
            plt.plot(xVals, y1, color="black", label="Quadratic Model", linewidth=3)
            fBest = (
                fx
                + g * (bestX - x)
                + 0.5 * (g - gp + sigma1 * (x - xp)) / (x - xp) * (bestX - x) ** 2
            )[0]

        if "1_cubic" == kind:
            plt.plot(
                [xp[0] - dx, xp[0] + dx],
                [fxp - gp * dx, fxp + gp * dx],
                "-.",
                color="darkgreen",
                zorder=-1,
            )
            fBest = (
                fx
                + g * (bestX - x)
                + 0.5 * (g - gp + sigma1 * (x - xp)) / (x - xp) * (bestX - x) ** 2
                + c * np.abs(bestX - x) ** 3
            )[0]

    if "2" == kind:
        plt.plot(xVals, y2, color="black", label="Quadratic Model", linewidth=3)
        fBest = (
            fx
            + g * (bestX - x)
            + 0.5 * (g - gp + sigma2 * (x - xp)) / (x - xp) * (bestX - x) ** 2
        )[0]

    if "3" == kind:
        plt.plot(xVals, y1, ":", color="black", label="Quadratic Model", linewidth=3)
        plt.plot(xVals, y3, color="black", label="Clipped Model", linewidth=3)
        plt.errorbar(
            xp,
            fxp,
            yerr=[(fxp - fx) / 2, (fxp - fx)],
            fmt="o",
            color="darkgreen",
            capsize=10,
            linewidth=3,
            zorder=-1,
        )
        fBest = (
            fx
            + g * (bestX - x)
            + 0.5 * (g - gp + sigma3 * (x - xp)) / (x - xp) * (bestX - x) ** 2
        )[0]

    plt.arrow(
        bestX,
        bestY,
        0,
        fBest - bestY,
        head_width=0.05,
        head_length=0.05,
        length_includes_head=True,
        fc="red",
        ec="red",
        zorder=5,
    )

    plt.scatter([xp], [fxp], color="black", zorder=5)
    plt.text(
        xp[0] + (-0.6 if "3" == kind else +0.1),
        fxp - 0.05,
        "$x_k$",
        fontsize=30,
        color="black",
    )
    plt.scatter([x], [fx], color="black", zorder=5)
    plt.text(x[0] - 0.1, fx - 0.15, "$x_{k+1}$", fontsize=30, color="black")

    plt.axis("off")

    plt.legend(loc="upper left", fontsize=20)

    plt.ylim(
        yVals.min() - 0.15 * (yVals.max() - yVals.min()),
        yVals.max() + 0.05 * (yVals.max() - yVals.min()),
    )
    plt.savefig(
        f"doc_private/imgs/modified_secant/trial_{kind}.pdf",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    a = 2.0 * 0.05
    b = 1.0 * 0.05
    c = 1.0 * 0.05
    d = 0.0 * 0.05
    e = 2.0 * 0.05

    def f(xi):
        return e * xi**4 + a * xi**3 + b * xi**2 + c * xi + d

    def f_prime(xi):
        return 4 * e * xi**3 + 3 * a * xi**2 + 2 * b * xi + c

    def f_double_prime(xi):
        return 12 * e * xi**2 + 6 * a * xi + 2 * b

    # def f(xi):
    #     return xi**10

    # def f_prime(xi):
    #     return 10 * xi**9

    # def f_double_prime(xi):
    #     return 90 * xi**8

    for kind in ["EXPLAIN", "HESS", "BFGS", "2", "1_cubic", "1_quadratic", "3"]:
        trial(
            x_=0.0,
            xp_=1.0,
            f=f,
            f_prime=f_prime,
            f_double_prime=f_double_prime,
            xMin=-1.2,
            xMax=+1.5,
            kind=kind,
        )
