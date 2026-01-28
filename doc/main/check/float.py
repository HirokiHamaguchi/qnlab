import os
import subprocess
import warnings
from pathlib import Path

import numpy as np

from qnlab.problem.cutest import CUTEstQNProblem


def compute_feps(precision: int) -> float:
    """Compute \feps using CUTEstQNProblem.get_machine_eps with zero noise."""

    dummy = CUTEstQNProblem.__new__(CUTEstQNProblem)
    dummy.precision = precision
    return float(dummy.get_machine_eps())


def copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard using xclip or other available tool."""
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"], input=text.encode(), check=True
        )
        print("\n✓ Table copied to clipboard!")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n✗ Could not copy to clipboard (xclip not found)")


def format_to_latex(value) -> str:
    """Format a number for LaTeX output."""
    if np.isinf(value):
        return r"\mathrm{nan}"

    formatted = np.format_float_scientific(value, precision=2, unique=False)
    return (
        formatted.replace("e-0", " \\times 10^{-").replace("e-", " \\times 10^{-") + "}"
    )


warnings.filterwarnings("ignore", category=RuntimeWarning)

formats = [
    ("64-bit double precision", compute_feps(64), np.float64),
    ("32-bit single precision", compute_feps(32), np.float32),
    ("16-bit half precision", compute_feps(16), np.float16),
]

# Build the table content
table_rows = []
for format_name, feps_value, dtype in formats:
    info = np.finfo(dtype)  # type: ignore
    eps, nmant, tiny = info.eps, info.nmant, info.tiny

    assert eps == dtype(2) ** (-nmant)  # type: ignore

    tiny_str = format_to_latex(tiny)
    eps_str = format_to_latex(eps)
    feps_str = format_to_latex(feps_value)

    table_rows.append(
        f"        {format_name:<23} & ${tiny_str}$ & $2^{{-{nmant}}} \\approx {eps_str}$ & ${feps_str}$ \\\\"
    )

# Get the relative path of this script
script_path = os.path.relpath(__file__)

# Construct the full LaTeX table
latex_output = f"""% !!! Auto-generated table by {script_path}
\\begin{{table}}[t]
    \\centering
    \\caption{{
        The ``min abs value'' is the smallest positive normalized number,
        the ``relative error'' is the maximum relative rounding error,
        and $\\feps$ is the parameter in \\eqref{{eq:relative_error_model}} used in our experiments.
    }}
    \\begin{{tabular}}{{l|ccc}}
        \\toprule
        explanation             & min abs value           & relative error                         & $\\feps$                  \\\\
        \\midrule
{chr(10).join(table_rows)}
        \\bottomrule
    \\end{{tabular}}
    \\label{{tab:machine_epsilon}}
\\end{{table}}
% !!! End of auto-generated table"""

output_path = Path(__file__).with_name("machine_epsilon_table.tex")
output_path.write_text(latex_output + "\n", encoding="utf-8")

print(latex_output)
print(f"\n✓ Saved LaTeX table to {output_path}")
copy_to_clipboard(latex_output)
