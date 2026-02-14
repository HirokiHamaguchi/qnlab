import numpy as np
import matplotlib.pyplot as plt

from qnlab.util.doc_paths import doc_imgs_dir

OUTPUT_DIR = doc_imgs_dir("quasi_newton")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


def f(x):
    return np.sin(x + 5)


def f_prime(x):
    return np.cos(x + 5)


L = 1.0


x = np.linspace(-2, 2.5, 400)
f_values = f(x)
f_prime_values = f_prime(x)

x0 = 0.0
f0 = f(x0)
f_prime0 = f_prime(x0)

descent_upper = f0 + f_prime0 * (x - x0) + 0.5 * L * (x - x0) ** 2
lip_upper = f_prime0 + L * (x - x0)
lip_lower = f_prime0 - L * (x - x0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

axes[0].plot(x, f_values)
axes[0].plot(x, descent_upper, linestyle="--")
axes[0].scatter([x0], [f0], zorder=5)
axes[0].set_title("$L$-smoothness of $f$", fontsize=16)
min_0 = min(np.min(f_values), np.min(descent_upper))
max_0 = max(np.max(f_values), np.max(descent_upper))
print(f"f(x) range: [{min_0:.2f}, {max_0:.2f}]")

axes[1].plot(x, f_prime_values)
axes[1].fill_between(x, lip_lower, lip_upper, color="gray", alpha=0.3, edgecolor="none")
axes[1].scatter([x0], [f_prime0], zorder=5)
axes[1].set_title("Lipschitz Continuity of $\\nabla f$", fontsize=16)
min_1 = min(np.min(f_prime_values), np.min(lip_lower), np.min(lip_upper))
max_1 = max(np.max(f_prime_values), np.max(lip_lower), np.max(lip_upper))
print(f"f'(x) range: [{min_1:.2f}, {max_1:.2f}]")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

axes[0].annotate(
    "$f(x)$",
    xy=(-2.2, min_0),
    xytext=(-2.2, max_0 - 0.05 * (max_0 - min_0)),
    arrowprops=dict(arrowstyle="<-", color="black"),
    ha="center",
    va="bottom",
    fontsize=12,
)
axes[1].annotate(
    "$f'(x)$",
    xy=(-2.2, min_1),
    xytext=(-2.2, max_1 - 0.05 * (max_1 - min_1)),
    arrowprops=dict(arrowstyle="<-", color="black"),
    ha="center",
    va="bottom",
    fontsize=12,
)

fig.savefig(OUTPUT_DIR / "l-smooth.pdf", bbox_inches="tight")
