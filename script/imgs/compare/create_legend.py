import os
import shutil
import math
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qnlab-matplotlib")
)

import matplotlib.pyplot as plt

from qnlab.util.method import COLORS, LINE_STYLES

LEGENDS = {
    "_legend.pdf": [
        ("NTRQN", "Ours"),
        ("NTRQN-MS", "Ours-MS"),
        ("NTRQN-SP", "Ours-SP"),
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
        ("NTRQN-SP", "Ours-SP"),
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
        ("NTRQN-SP", "Ours-SP"),
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
        ("NTRQN-SP", "Ours-SP"),
        ("NTRQN-Restart", "Ours-R"),
        ("NTRQN-MS-Restart", "Ours-MS-R"),
    ],
    "_legend_restart.pdf": [
        ("NTRQN", "Ours (no restart)"),
        ("NTRQN-MS", "Ours-MS (no restart)"),
        ("NTRQN-Restart", "Ours-R (restart)"),
        ("NTRQN-MS-Restart", "Ours-MS-R (restart)"),
    ],
}


def create_legend(filename, entries):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": shutil.which("latex") is not None,
            "font.family": "serif",
            "font.size": 20,
            "figure.dpi": 300,
            "lines.linewidth": 2.0,
        }
    )

    ncol = min(len(entries), 5)
    nrows = math.ceil(len(entries) / ncol)
    if nrows > 1:
        # Matplotlib fills legend columns first; interleave the rows so that
        # the displayed entries instead follow the input order from left to right.
        entries = [
            entries[row * ncol + col]
            for col in range(ncol)
            for row in range(nrows)
            if row * ncol + col < len(entries)
        ]
    alg_names = [name for name, _ in entries]

    fig, ax = plt.subplots(figsize=(13.5, 0.8 + 0.7 * nrows))
    handles = []
    for name in alg_names:
        color = COLORS[name]
        linestyle = LINE_STYLES[name]
        (handle,) = ax.plot([], [], linestyle, color=color, linewidth=3.0, markersize=10)
        handles.append(handle)

    legend_names = [display_name for _, display_name in entries]
    ax.legend(
        handles,
        legend_names,
        loc="center",
        bbox_to_anchor=(0, 0, 1, 1),
        mode="expand",
        borderaxespad=0.05,
        framealpha=0.98,
        edgecolor="black",
        fancybox=True,
        fontsize=22,
        ncol=ncol,
        frameon=True,
    )
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    out_dir = Path(__file__).parent.parent.parent.parent / "doc" / "imgs" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    legend_path = out_dir / filename
    fig.savefig(
        legend_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300,
    )
    plt.close(fig)
    print(f"Saved legend to {legend_path}")


if __name__ == "__main__":
    for filename, entries in LEGENDS.items():
        create_legend(filename, entries)
