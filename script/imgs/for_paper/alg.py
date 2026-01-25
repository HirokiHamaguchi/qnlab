import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

from qnlab.util.doc_paths import doc_imgs_dir

OUTPUT_DIR = doc_imgs_dir("for_paper")

# -------------------------
# LaTeX settings
# -------------------------
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# -------------------------
# data (single sequence)
# -------------------------
k = np.arange(0, 9)
f = np.array([8.0, 6.2, 5.0, 5.4, 5.2, 3.8, 3.2, 3.6, 3.2])

# index sets
K0 = {0, 1, 2, 5, 6}  # blue
Kp = {3, 4, 7, 8}  # red

# -------------------------
# figure
# -------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(10, 4.8))

# draw line segments with color switching
for i in range(len(k) - 1):
    if k[i] in K0:
        color = "#1f77b4"  # blue
    else:
        color = "#d62728"  # red

    plt.plot(k[i : i + 2], f[i : i + 2], "-o", color=color, lw=2.5, ms=6)

# final point
last_color = "#1f77b4" if k[-1] in K0 else "#d62728"
plt.plot(k[-1], f[-1], "o", color=last_color, ms=6)

# continuation hint (dashed line after k=8)
plt.plot([8, 8.5], [3.2, 3.0], "--", color="#d62728", lw=2.5)

# -------------------------
# annotations (inequalities)
# -------------------------
x = 2.1
y = 4.2
plt.text(
    x,
    y,
    r"$\bar f(x_{k+1}) \leq \bar f(x_k)+\Delta_k$",
    fontsize=25,
    ha="center",
    bbox=dict(facecolor="white"),
)
plt.text(
    x,
    y - 0.55,
    r"$(k \in K)$",
    fontsize=20,
    ha="center",
    bbox=dict(facecolor="white"),
)

x = 6.5
y = 5.3
plt.text(
    x,
    y,
    r"$\bar f(x_k)\leq \bar f(x_j)-\Delta_{j}$",
    fontsize=25,
    ha="center",
    bbox=dict(facecolor="white"),
)
plt.text(
    x,
    y - 0.55,
    r"$(k \in K^0, \, 0 \leq j < k)$",
    fontsize=20,
    ha="center",
    bbox=dict(facecolor="white"),
)


arrow_h = FancyArrowPatch(
    (2.0, f[2]), (5.0, f[2]), arrowstyle="-", lw=1, linestyle="dashed", color="gray"
)
plt.gca().add_patch(arrow_h)

arrow_v1 = FancyArrowPatch(
    (3.0, f[2]), (3.0, f[3]), arrowstyle="->", mutation_scale=20, lw=2, color="black"
)
plt.gca().add_patch(arrow_v1)

arrow_v2 = FancyArrowPatch(
    (5.0, f[2]), (5.0, f[5]), arrowstyle="->", mutation_scale=30, lw=2, color="black"
)
plt.gca().add_patch(arrow_v2)

# -------------------------
# legend (manual)
# -------------------------
plt.plot(
    [],
    [],
    "-o",
    color="#1f77b4",
    label=r"\makebox[0pt][l]{$k\in K^0$}\phantom{$k\in K^+$} $(\mu_k=0)$",
)
plt.plot(
    [],
    [],
    "-o",
    color="#d62728",
    label=r"\makebox[0pt][l]{$k\in K^+$}\phantom{$k\in K^+$} $(\mu_k>0)$",
)
plt.legend(frameon=True, fontsize=25)

# -------------------------
# axes & output
# -------------------------
plt.xlabel(r"$k$", fontsize=25)
plt.ylabel(r"$\bar f(x_k)$", fontsize=25)
plt.tick_params(labelsize=15)
plt.yticks([])
plt.xlim(-0.1, 8.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "alg.pdf")
plt.close()
