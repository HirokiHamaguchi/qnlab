import matplotlib.pyplot as plt
import numpy as np

from qnlab.util.doc_paths import doc_imgs_dir

OUTPUT_DIR = doc_imgs_dir("quasi_newton")

# Seaborn-like "deep" palette colors
COLOR_FUNCTION = "#4C72B0"
COLOR_ARMIJO = "#DD8452"
# light green and green
COLOR_CURVATURE = "#55C667"
COLOR_WOLFE = "#009E73"

a = 1 / 3 * 1.2**3
b = -1 / 2 * 1.2**2
c = -2 * 1.2

x = np.linspace(-0.5, 2.8, 400)
f = a * x**3 + b * x**2 + c * x
x0 = -0.1
f0 = a * x0**3 + b * x0**2 + c * x0
grad0 = 3 * a * x0**2 + 2 * b * x0 + c
grad = 3 * a * x**2 + 2 * b * x + c
c1 = 0.3
c2 = 0.5

armijo_line = f0 + c1 * grad0 * (x - x0)

plt.figure(figsize=(7, 4))
plt.plot(x, f, color=COLOR_FUNCTION)
plt.plot(x, armijo_line, linestyle="--", color=COLOR_ARMIJO)
plt.scatter([x0], [f0], color=COLOR_FUNCTION, zorder=5)

# Armijo条件を満たす範囲
armijo_mask = f <= armijo_line
armijo_range = x[armijo_mask]

# Curvature condition
curvature_mask = grad >= c2 * grad0
curvature_range = x[curvature_mask]

tangent_x0 = np.linspace(x0 - 0.3, x0 + 0.3, 10)
tangent_y0 = grad0 * (tangent_x0 - x0) + f0
plt.plot(tangent_x0, tangent_y0, linestyle="--", color=COLOR_CURVATURE)

min_curv_x = curvature_range[0]
tangent_x = np.linspace(min_curv_x - 0.3, min_curv_x + 0.3, 10)
tangent_y = grad[min_curv_x == x] * (tangent_x - min_curv_x) + f[min_curv_x == x]
plt.plot(tangent_x, tangent_y, linestyle="--", color=COLOR_CURVATURE)


# Wolfe条件
wolfe_mask = armijo_mask & curvature_mask
wolfe_range = x[wolfe_mask]

y_arrow = min(f) - 0.5

plt.annotate(
    "",
    xy=(armijo_range[-1], y_arrow),
    xytext=(armijo_range[0], y_arrow),
    arrowprops=dict(
        arrowstyle="<->", color=COLOR_ARMIJO, linewidth=2, shrinkA=0, shrinkB=0
    ),
)
plt.annotate(
    "",
    xy=(curvature_range[-1], y_arrow - 0.5),
    xytext=(curvature_range[0], y_arrow - 0.5),
    arrowprops=dict(
        arrowstyle="<->", color=COLOR_CURVATURE, linewidth=2, shrinkA=0, shrinkB=0
    ),
)
plt.annotate(
    "",
    xy=(wolfe_range[-1], y_arrow - 1.0),
    xytext=(wolfe_range[0], y_arrow - 1.0),
    arrowprops=dict(
        arrowstyle="<->", color=COLOR_WOLFE, linewidth=2, shrinkA=0, shrinkB=0
    ),
)

plt.text(
    -0.5, y_arrow, "Armijo", color=COLOR_ARMIJO, fontsize=15, ha="center", va="center"
)
plt.text(
    -0.5,
    y_arrow - 0.5,
    "Curvature",
    color=COLOR_CURVATURE,
    fontsize=15,
    ha="center",
    va="center",
)
plt.text(
    -0.5, y_arrow - 1, "Wolfe", color=COLOR_WOLFE, fontsize=15, ha="center", va="center"
)

plt.xticks([])
plt.yticks([])
plt.xlim(-0.5 - 0.5, 2.8 + 0.5)
plt.ylim(min(f) - 1.5, f0 + 1)
plt.box(False)
plt.title("Armijo and (weak) Wolfe Conditions", fontsize=15)
plt.savefig(OUTPUT_DIR / "armijo_wolfe_conditions.pdf", bbox_inches="tight")
plt.close()
