import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.chdir(Path(__file__).parent)


class StronglyConvexFunction:
    def __init__(self, mu):
        self.mu = mu

    def f(self, x):
        return np.log(1 + np.exp(x)) - x / 2 + (self.mu / 2) * x**2

    def df(self, x):
        return np.exp(x) / (1 + np.exp(x)) - 0.5 + self.mu * x

    def d2f(self, x):
        return np.exp(x) / (1 + np.exp(x)) ** 2 + self.mu

    def vis(self, x0):
        xs = [x0]
        for _ in range(30):
            x_next = xs[-1] - self.df(xs[-1]) / self.d2f(xs[-1])
            xs.append(x_next)

        f_points = self.f(np.array(xs))
        abs_max = max(abs(np.array(xs)))
        x = np.linspace(-abs_max * 1.2, abs_max * 1.2, 400)
        fx = self.f(x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, fx, label=r"$f(x)=\log(1+e^x)-x/2 + \mu x^2/2$")
        plt.title(rf"Graph of $f(x)$ with $\mu$={self.mu} and $x_0$={x0}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)

        plt.scatter(xs, f_points, color="red", label="Newton steps")
        for i in range(len(xs) - 1):
            plt.annotate(
                "",
                xy=(xs[i + 1], f_points[i + 1]),
                xytext=(xs[i], f_points[i]),
                arrowprops=dict(
                    arrowstyle="->", color="red", lw=1.5, mutation_scale=30
                ),
            )

        plt.legend(loc="upper center")
        plt.savefig(f"newton_failure_strongly_convex_function_{self.mu}_{x0}.pdf")
        plt.close()


class SqrtFunction:
    def __init__(self):
        pass

    def f(self, x):
        return np.sqrt(1 + x**2)

    def df(self, x):
        return x / np.sqrt(1 + x**2)

    def d2f(self, x):
        return 1 / (1 + x**2) ** (3 / 2)

    def vis(self, x0):
        xs = [x0]
        for _ in range(3):
            x_next = xs[-1] - self.df(xs[-1]) / self.d2f(xs[-1])
            xs.append(x_next)

        f_points = self.f(np.array(xs))
        abs_max = max(abs(np.array(xs)))
        x = np.linspace(-abs_max * 1.2, abs_max * 1.2, 400)
        fx = self.f(x)

        plt.figure(figsize=(8, 5))
        plt.plot(x, fx, label=r"$f(x)=\sqrt{1+x^2}$")
        plt.title(rf"Graph of $f(x)$ with $x_0$={x0}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)

        plt.scatter(xs, f_points, color="red", label="Newton steps")
        for i in range(len(xs) - 1):
            plt.annotate(
                "",
                xy=(xs[i + 1], f_points[i + 1]),
                xytext=(xs[i], f_points[i]),
                arrowprops=dict(
                    arrowstyle="->", color="red", lw=1.5, mutation_scale=30
                ),
            )

        plt.legend(loc="upper center")
        plt.savefig(f"newton_failure_sqrt_function_{x0}.pdf")
        plt.close()


if __name__ == "__main__":
    for mu_val, x0_val in [(0.1, -4), (0.01, -4)]:
        func: StronglyConvexFunction = StronglyConvexFunction(mu_val)
        func.vis(x0_val)

    for x0_val2 in [1.1]:
        func2: SqrtFunction = SqrtFunction()
        func2.vis(x0_val2)
