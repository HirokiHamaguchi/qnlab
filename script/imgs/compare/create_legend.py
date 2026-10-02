import math
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qnlab-matplotlib")
)

import matplotlib.pyplot as plt

from qnlab.util.method import COLORS, LINE_STYLES, SENSITIVITY_COLORS

LEGENDS = {
    "_legend.pdf": [
        ("NTRQN-MS", "Ours"),
        ("NTRQN-Restart", "Ours-AR"),
        ("Line", "Line"),
        ("Reg", "Reg"),
        ("Reg-Sec", "Reg-Sec"),
        ("SciPy", "SciPy"),
        ("NTQN", "NTQN"),
    ],
    "_legend_noise.pdf": [
        ("NTRQN-MS", "Ours"),
        ("NTRQN-Restart", "Ours-AR"),
        ("Line", "Line"),
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
        ("NTRQN-MS", "Ours"),
        ("NTRQN-Restart", "Ours-AR"),
        ("Line", "Line"),
        ("Reg", "Reg"),
        ("Reg-Sec", "Reg-Sec"),
        ("SciPy", "SciPy"),
        ("NTQN", "NTQN"),
        ("ASTR1-Adagrad", "ASTR1-Adagrad"),
    ],
    "_legend_restart.pdf": [
        ("NTRQN-MS", "Ours"),
        ("NTRQN-Restart", "Ours-AR"),
    ],
}


def column_major_entries(entries, ncol):
    """Convert left-to-right row order to Matplotlib's column-first order."""
    if not entries or not 1 <= ncol <= len(entries):
        raise ValueError("Legend requires entries and a valid column count")
    nrows = math.ceil(len(entries) / ncol)
    return [
        entries[row * ncol + col]
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(entries)
    ]


def draw_legend(filename, entries, ncol, fontsize=22):
    """Draw (label, color, line style) entries supplied in row order."""
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

    nrows = math.ceil(len(entries) / ncol)
    entries = column_major_entries(entries, ncol)

    fig, ax = plt.subplots(figsize=(13.5, 0.8 + 0.7 * nrows))
    handles = []
    for _, color, linestyle in entries:
        (handle,) = ax.plot(
            [], [], linestyle, color=color, linewidth=3.0, markersize=10
        )
        handles.append(handle)

    legend_names = [label for label, _, _ in entries]
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
        fontsize=fontsize,
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


def create_legend(filename, entries):
    # Eight-method comparisons use two complete rows of four.
    draw_legend(
        filename,
        [(label, COLORS[name], LINE_STYLES[name]) for name, label in entries],
        ncol=min(len(entries), 4),
    )


def create_sensitivity_legend():
    entries = []
    # One method per row, with error settings ordered left to right.
    for method, source, marker in (
        ("Ours", "NTRQN-MS", "o"),
        ("Ours-AR", "NTRQN-Restart", "s"),
    ):
        for setting, setting_style in (
            ("underestimated", "--"),
            ("nominal", "-"),
            ("overestimated", ":"),
        ):
            entries.append(
                (
                    f"{method} ({setting})",
                    SENSITIVITY_COLORS[setting][source],
                    marker + setting_style,
                )
            )
    draw_legend("_legend_sensitivity.pdf", entries, ncol=3, fontsize=20)


if __name__ == "__main__":
    for filename, entries in LEGENDS.items():
        create_legend(filename, entries)
    create_sensitivity_legend()
