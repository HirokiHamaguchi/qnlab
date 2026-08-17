"""Generate the local image and PDF assets referenced by the SciPy issue draft."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "doc" / "SciPy"
INDIVIDUAL_DIR = REPO_ROOT / "doc" / "imgs" / "compare" / "individual"
LEGEND_PATH = REPO_ROOT / "doc" / "imgs" / "compare" / "_legend.pdf"
UNBOXED_ORACLE_DIR = INDIVIDUAL_DIR / "unboxed" / "precision64"
UNBOXED_TIME_DIR = INDIVIDUAL_DIR / "unboxed" / "time" / "precision64"


def render_first_page(source: Path, destination: Path, dpi: int = 240) -> None:
    """Render the first page of a PDF as a PNG for Markdown display."""
    with fitz.open(source) as document:
        page = document[0]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(destination)


def latex_escape(value: str) -> str:
    """Escape the subset of TeX special characters used in problem names."""
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def relative_tex_path(path: Path) -> str:
    """Return a POSIX path to an asset, relative to the generated TeX file."""
    return Path(os.path.relpath(path, OUTPUT_DIR)).as_posix()


def paired_plot_block(name: str, oracle_path: Path, time_path: Path) -> str:
    """Place the oracle-call and wall-clock plots for one problem together."""
    oracle_relative_path = relative_tex_path(oracle_path)
    time_relative_path = relative_tex_path(time_path)
    escaped_name = latex_escape(name)
    return rf"""
\begin{{minipage}}[t]{{\textwidth}}
  \centering
  \texttt{{\bfseries {escaped_name}}}\par\smallskip
  \begin{{minipage}}[t]{{0.47\textwidth}}
    \centering\scriptsize Oracle calls\par
    \includegraphics[width=\linewidth,height=0.20\textheight,keepaspectratio]{{{oracle_relative_path}}}
  \end{{minipage}}\hfill
  \begin{{minipage}}[t]{{0.47\textwidth}}
    \centering\scriptsize Wall-clock time\par
    \includegraphics[width=\linewidth,height=0.20\textheight,keepaspectratio]{{{time_relative_path}}}
  \end{{minipage}}
\end{{minipage}}
""".strip()


def paired_plots(oracle_directory: Path, time_directory: Path) -> str:
    """Group plots by problem, with both metrics on each row."""
    oracle_plots = {path.stem: path for path in oracle_directory.glob("*.pdf")}
    time_plots = {path.stem: path for path in time_directory.glob("*.pdf")}
    if not oracle_plots:
        raise FileNotFoundError(f"No PDF plots found in {oracle_directory}")
    if oracle_plots.keys() != time_plots.keys():
        oracle_only = sorted(oracle_plots.keys() - time_plots.keys())
        time_only = sorted(time_plots.keys() - oracle_plots.keys())
        raise ValueError(
            "Oracle-call and wall-clock plots do not cover the same problems: "
            f"oracle only={oracle_only}, time only={time_only}"
        )

    names = sorted(oracle_plots, key=str.casefold)
    chunks = [rf"\noindent {len(names)} problems.\par\smallskip"]
    for index, name in enumerate(names, start=1):
        chunks.append(paired_plot_block(name, oracle_plots[name], time_plots[name]))
        if index != len(names):
            chunks.append(r"\par\smallskip")
    return "\n".join(chunks)


def generate_individual_results_tex(destination: Path) -> None:
    """Create a contact-sheet TeX document containing every individual plot."""
    legend_path = relative_tex_path(LEGEND_PATH)
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\pagestyle{fancy}
\fancyhf{}
\lhead{CUTEst individual results}
\rhead{\thepage}
\begin{document}
\title{CUTEst individual optimization traces}
\author{Generated from the qnlab experiment artifacts}
\date{}
\maketitle
\begin{abstract}
This appendix collects all per-problem plots used to audit the aggregate
performance profiles in the accompanying SciPy issue draft.  Oracle-call and
wall-clock plots for each of the 220 unconstrained, 64-bit CUTEst problems are
shown side by side.  These plots show heterogeneous solver behavior and are
evidence against interpreting an aggregate profile as a claim that one solver
wins on every problem.
\end{abstract}
\begin{center}
  \textbf{Solver legend}\par\smallskip
  \includegraphics[width=0.94\textwidth,height=0.16\textheight,keepaspectratio]{LEGEND_PATH_PLACEHOLDER}
\end{center}
""".replace("LEGEND_PATH_PLACEHOLDER", legend_path)
    body = paired_plots(UNBOXED_ORACLE_DIR, UNBOXED_TIME_DIR)
    destination.write_text(preamble + body + "\n\\end{document}\n", encoding="utf-8")


def compile_tex(source: Path) -> None:
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            source.name,
        ],
        cwd=source.parent,
        check=True,
    )
    subprocess.run(
        ["latexmk", "-c", source.name],
        cwd=source.parent,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-compile", action="store_true", help="Generate TeX without running latexmk"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_first_page(
        REPO_ROOT / "doc" / "imgs" / "for_paper" / "time.pdf",
        OUTPUT_DIR / "time_per_iteration.png",
    )

    tex_path = OUTPUT_DIR / "cutest_individual_results.tex"
    generate_individual_results_tex(tex_path)
    if not args.no_compile:
        compile_tex(tex_path)


if __name__ == "__main__":
    main()
