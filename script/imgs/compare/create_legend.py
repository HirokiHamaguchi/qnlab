from pathlib import Path

import matplotlib.pyplot as plt

from qnlab.util.method import COLORS, LINE_STYLES


LEGENDS = {
    "_legend.pdf": [
        ("NTRQN", "Ours"),
        ("NTRQN-MS", "Ours-MS"),
        ("Line", "Line"),
        ("Line-MS", "Line-MS"),
        ("Reg", "Reg"),
        ("Reg-Sec", "Reg-Sec"),
        ("SciPy", "SciPy"),
        ("NTQN", "NTQN"),
    ],
    "_legend_noise.pdf": [
        ("NTRQN", "Ours"),
        ("NTRQN-MS", "Ours-MS"),
        ("Line", "Line"),
        ("Line-MS", "Line-MS"),
        ("Reg", "Reg"),
        ("Reg-Sec", "Reg-Sec"),
        ("SciPy", "SciPy"),
        ("NTQN", "NTQN"),
        ("ASTR1-Adagrad", "ASTR1-Adagrad"),
    ],
    "_legend_ntqn_termination.pdf": [
        ("NTQN", "NTQN (common stop)"),
        ("NTQN-Default-Termination", "NTQN (recommended stop)"),
    ],
    "_legend_precision.pdf": [
        ("NTRQN", "Ours"),
        ("NTRQN-MS", "Ours-MS"),
        ("Line", "Line"),
        ("Line-MS", "Line-MS"),
        ("Reg", "Reg"),
        ("Reg-Sec", "Reg-Sec"),
        ("SciPy", "SciPy"),
        ("NTQN", "NTQN"),
        ("ASTR1-Adagrad", "ASTR1-Adagrad"),
    ],
    "_legend_sensitivity.pdf": [
        ("NTRQN", "Ours"),
        ("NTRQN-MS", "Ours-MS"),
    ],
    "_legend_restart.pdf": [
        ("NTRQN", "Ours (no restart)"),
        ("NTRQN-Restart", "Ours-R (restart)"),
    ],
}


def create_legend(filename, entries):
    alg_names = [name for name, _ in entries]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 20,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )

    ncol = 5 if len(entries) > 5 else len(entries)
    fig_width = 11 if len(entries) > 5 else max(4, 2.6 * len(entries))
    fig, ax = plt.subplots(figsize=(fig_width, 1.45))
    handles = []
    for name in alg_names:
        color = COLORS[name]
        linestyle = LINE_STYLES[name]
        (handle,) = ax.plot(
            [], [], linestyle, color=color, linewidth=2.5, markersize=8
        )
        handles.append(handle)

    legend_names = [display_name for _, display_name in entries]
    ax.legend(
        handles,
        legend_names,
        loc="center",
        framealpha=0.98,
        edgecolor="black",
        fancybox=True,
        fontsize=14,
        ncol=ncol,
        frameon=True,
    )
    ax.axis("off")
    plt.tight_layout()

    out_dir = Path(__file__).parent.parent.parent.parent / "doc" / "imgs" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_path = out_dir / filename
    fig.savefig(legend_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved legend to {legend_path}")


if __name__ == "__main__":
    for filename, entries in LEGENDS.items():
        create_legend(filename, entries)
