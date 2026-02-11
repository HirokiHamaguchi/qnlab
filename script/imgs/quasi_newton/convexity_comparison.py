import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from qnlab.util.doc_paths import doc_imgs_dir

OUTPUT_DIR = doc_imgs_dir("quasi_newton")

sns.set_style("darkgrid")

# Set font to match LaTeX
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Set font sizes globally
plt.rcParams.update(
    {
        "font.size": 20,  # labels
        "figure.titlesize": 30,  # figure title
        "axes.titlesize": 30,  # title
        "axes.labelsize": 20,  # x and y labels
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 20,  # legend
    }
)


# Define functions and their second derivatives
def f_x4(x):
    """f(x) = x^4 + x"""
    return x**4 + x


def f2_x4(x):
    """f''(x) = 12x^2 (Hessian at x=0 is 0)"""
    return 12 * x**2


def quad_x4(x):
    """Quadratic approximation at x=0: q(x) = x"""
    return x


def f_exp(x):
    """f(x) = e^x"""
    return np.exp(x)


def f2_exp(x):
    """f''(x) = e^x (Hessian at x=0 is 1)"""
    return np.exp(x)


def quad_exp(x):
    """Quadratic approximation at x=0: q(x) = 1 + x + 0.5*x^2"""
    return 1 + x + 0.5 * x**2


def f_x4_x2(x):
    """f(x) = x^4 + x^2"""
    return x**4 + x**2


def f2_x4_x2(x):
    """f''(x) = 12x^2 + 2 (Hessian at x=0 is 2)"""
    return 12 * x**2 + 2


def quad_x4_x2(x):
    """Quadratic approximation at x=0: q(x) = x^2"""
    return x**2


def f_cosh(x):
    """f(x) = cosh(x)"""
    return np.cosh(x)


def f2_cosh(x):
    """f''(x) = cosh(x) (Hessian at x=0 is 1)"""
    return np.cosh(x)


def quad_cosh(x):
    """Quadratic approximation at x=0: q(x) = 1 + 0.5*x^2"""
    return 1 + 0.5 * x**2


x_range = np.linspace(-1.5, 1.5, 400)

# ===== Figure 1: Convex but not strongly convex =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.set_title(r"$f(x) = x^4 + x$", fontsize=20, pad=20)
ax.plot(x_range, f_x4(x_range), color="tab:blue", linewidth=2)
ax.plot(x_range, quad_x4(x_range), "--", color="tab:orange", linewidth=2)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title(r"$f(x) = e^x$", fontsize=20, pad=20)
ax.plot(x_range, f_exp(x_range), color="tab:blue", linewidth=2)
ax.plot(x_range, quad_exp(x_range), "--", color="tab:orange", linewidth=2)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-1)

fig.suptitle("Convex but not strongly convex")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "convexity_comparison_convex.pdf")
plt.close()

# ===== Figure 2: Strongly convex =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.set_title(r"$f(x) = x^4 + x^2$", fontsize=20, pad=20)
ax.plot(x_range, f_x4_x2(x_range), color="tab:blue", linewidth=2)
ax.plot(x_range, quad_x4_x2(x_range), "--", color="tab:orange", linewidth=2)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title(r"$f(x) = \cosh(x) = \frac{e^x + e^{-x}}{2}$", fontsize=20, pad=20)
ax.plot(x_range, f_cosh(x_range), color="tab:blue", linewidth=2)
ax.plot(x_range, quad_cosh(x_range), "--", color="tab:orange", linewidth=2)
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.grid(True, alpha=0.3)

fig.suptitle("Strongly convex")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "convexity_comparison_strongly_convex.pdf")
plt.close()
