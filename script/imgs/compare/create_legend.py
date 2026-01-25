from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from qnlab.util.method import get_methods


def create_legend():
    methods, ALGORITHM_COLORS, ALGORITHM_LINE_STYLES = get_methods()
    alg_names = [method.label for method, _ in methods]

    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 20,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )

    fig, ax = plt.subplots(figsize=(6, 1.5))
    handles = []
    for name in alg_names:
        color = ALGORITHM_COLORS.get(name, "black")
        linestyle = ALGORITHM_LINE_STYLES.get(name, "o-")
        (handle,) = plt.step(
            [0, 0], [0, 0], linestyle, color=color, linewidth=2.5, markersize=8
        )
        handles.append(handle)

    legend_names = [name.replace("Hamaguchi", "Ours") for name in alg_names]
    ax.legend(
        handles,
        legend_names,
        loc="center",
        framealpha=0.98,
        edgecolor="black",
        fancybox=True,
        fontsize=16,
        ncol=max(1, (len(alg_names) + 1) // 2),
        frameon=True,
    )
    ax.axis("off")
    plt.tight_layout()

    out_dir = Path(__file__).parent.parent.parent.parent / "doc" / "imgs" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_path = out_dir / "_legend.pdf"
    fig.savefig(legend_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved legend to {legend_path}")


if __name__ == "__main__":
    create_legend()
