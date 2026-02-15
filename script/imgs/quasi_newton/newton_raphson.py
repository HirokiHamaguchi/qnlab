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
        "axes.titlesize": 20,  # title
        "axes.labelsize": 20,  # x and y labels
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 20,  # legend
    }
)


# Define function and derivatives
def f(x):
    """Scalar- or array-compatible function f(x) = x^3 - 2 x^2 + x."""
    x = np.asarray(x)
    return x**3 - 2 * x**2 + x


def df(x):
    """First derivative f'(x) = 3 x^2 - 4 x + 1."""
    x = np.asarray(x)
    return 3 * x**2 - 4 * x + 1


def d2f(x):
    """Second derivative f''(x) = 6 x - 4."""
    x = np.asarray(x)
    return 6 * x - 4


# Newton in optimization
def newton_opt(x0, df, d2f, n_iter=3):
    xs = [x0]
    for _ in range(n_iter):
        x_new = xs[-1] - df(xs[-1]) / d2f(xs[-1])
        xs.append(x_new)
    return xs


# Newton for root finding (on gradient)
def newton_root(x0, g, dg, n_iter=3):
    xs = [x0]
    for _ in range(n_iter):
        x_new = xs[-1] - g(xs[-1]) / dg(xs[-1])
        xs.append(x_new)
    return xs


x0 = 1.5
xs_root = newton_root(x0, df, d2f)
xs_opt = newton_opt(x0, df, d2f)

x = np.linspace(0.7, 1.8, 400)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Create viridis colormap for iterations
n_iter = len(xs_root) - 1
colors = plt.get_cmap("plasma")(
    np.linspace(0.2, 0.8, n_iter)
)  # Use middle part of viridis for green tones

# --- Left: Newton's method in optimization ---
ax = axes[0]
ax.plot(x, f(x), label="$f(x)$", color="tab:blue")
for i in range(len(xs_opt) - 1):
    x_i, x_next = xs_opt[i], xs_opt[i + 1]
    # Quadratic approximation
    a = 0.5 * d2f(x_i)
    b = df(x_i) - d2f(x_i) * x_i
    c = f(x_i) - df(x_i) * x_i + 0.5 * d2f(x_i) * x_i**2
    quad = a * x**2 + b * x + c
    x_model_min = -b / (2 * a)
    y_model_min = a * x_model_min**2 + b * x_model_min + c
    ax.plot(x, quad, "--", color=colors[i], alpha=0.9)
    ax.plot([x_i], [f(x_i)], "o", color="tab:red", markersize=10)
    ax.text(x_i, f(x_i) + 0.2, f"$x_{i}$", ha="center", fontsize=20)
    ax.plot([x_next, x_next], [0, f(x_next)], ":", color="tab:gray")
ax.set_title("Optimization on $f(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.legend()

# --- Right: Root finding on gradient (Newton's method) ---
ax = axes[1]
ax.plot(x, df(x), label=r"$\nabla f(x)$", color="tab:blue")
for i in range(len(xs_root) - 1):
    x_i, x_next = xs_root[i], xs_root[i + 1]
    # Tangent line
    tangent = df(x_i) + d2f(x_i) * (x - x_i)
    ax.plot(x, tangent, "--", color=colors[i], alpha=0.9)
    ax.plot([x_i], [df(x_i)], "o", color="tab:red", markersize=10)
    ax.text(x_i, df(x_i) + 0.3, f"$x_{i}$", ha="center", fontsize=20)
    ax.plot([x_next, x_next], [0, df(x_next)], ":", color="tab:gray")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Root finding on $\\nabla f(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel("$\\nabla f(x)$")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "newton_raphson.pdf")
plt.close()


# --- Storyboard: optimization-only (3 frames) ---
def plot_opt_frame(step_idx, x, xs_opt, colors):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x, f(x), label="$f(x)$", color="tab:blue")

    x_i = xs_opt[step_idx]
    x_next = xs_opt[step_idx + 1]

    # Quadratic approximation around current point
    a = 0.5 * d2f(x_i)
    b = df(x_i) - d2f(x_i) * x_i
    c = f(x_i) - df(x_i) * x_i + 0.5 * d2f(x_i) * x_i**2
    quad = a * x**2 + b * x + c
    ax.plot(
        x, quad, "--", color=colors[step_idx], alpha=0.9, label=f"$m_{step_idx}^*(x)$"
    )

    x_quad_min = -b / (2 * a)
    y_quad_min = a * x_quad_min**2 + b * x_quad_min + c

    # Current point and next point
    ax.plot([x_i], [f(x_i)], "o", color="tab:red", markersize=10)
    ax.text(x_i, float(f(x_i)) + 0.2, f"$x_{step_idx}$", ha="center", fontsize=18)
    ax.plot([x_next], [f(x_next)], "o", color="tab:green", markersize=8)
    ax.annotate(
        "",
        xy=(x_quad_min, y_quad_min),
        xytext=(x_i, float(f(x_i))),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    ax.set_title(f"Newton's Method: step {step_idx}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


for step_idx in range(3):
    fig = plot_opt_frame(step_idx, x, xs_opt, colors)
    out_path = OUTPUT_DIR / f"newton_opt_step{step_idx}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
