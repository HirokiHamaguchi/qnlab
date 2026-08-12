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


def render_first_page(source: Path, destination: Path, dpi: int = 240) -> None:
    """Render the first page of a PDF as a PNG for Markdown display."""
    with fitz.open(source) as document:
        page = document[0]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        pixmap.save(destination)


def latex_escape(value: str) -> str:
    """Escape the subset of TeX special characters used in problem names."""
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def plot_block(path: Path) -> str:
    relative_path = Path(os.path.relpath(path, OUTPUT_DIR)).as_posix()
    name = latex_escape(path.stem)
    return rf"""
\begin{{minipage}}[t]{{0.485\textwidth}}
  \centering
  \includegraphics[width=\linewidth,height=0.39\textheight,keepaspectratio]{{{relative_path}}}
  \par\smallskip
  \texttt{{{name}}}
\end{{minipage}}
""".strip()


def section(title: str, directory: Path) -> str:
    plots = sorted(directory.glob("*.pdf"), key=lambda path: path.stem.casefold())
    if not plots:
        raise FileNotFoundError(f"No PDF plots found in {directory}")

    chunks = [rf"\section{{{title}}}", rf"\noindent {len(plots)} problems.\par\medskip"]
    for index, plot in enumerate(plots, start=1):
        chunks.append(plot_block(plot))
        if index % 2:
            chunks.append(r"\hfill")
        elif index % 4:
            chunks.append(r"\par\medskip")
        if index % 4 == 0 and index != len(plots):
            chunks.append(r"\clearpage")
    chunks.append(r"\clearpage")
    return "\n".join(chunks)


def generate_individual_results_tex(destination: Path) -> None:
    """Create a contact-sheet TeX document containing every individual plot."""
    preamble = r"""\documentclass[10pt,a4paper,landscape]{article}
\usepackage[margin=10mm,headheight=14pt]{geometry}
\usepackage{graphicx}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\setlength{\parindent}{0pt}
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
performance profiles in the accompanying SciPy issue draft.  The first two
sections contain the same 220 unconstrained, 64-bit CUTEst problems plotted
against oracle calls and wall-clock time.  The final section contains 124
box-constrained, 64-bit problems plotted against oracle calls.  These plots
show heterogeneous solver behavior and are evidence against interpreting an
aggregate profile as a claim that one solver wins on every problem.
\end{abstract}
\tableofcontents
\clearpage
"""
    body = "\n".join(
        [
            section(
                "Unconstrained, 64-bit: oracle calls",
                INDIVIDUAL_DIR / "unboxed" / "precision64",
            ),
            section(
                "Unconstrained, 64-bit: wall-clock time",
                INDIVIDUAL_DIR / "unboxed" / "time" / "precision64",
            ),
            section(
                "Box-constrained, 64-bit: oracle calls",
                INDIVIDUAL_DIR / "boxed" / "precision64",
            ),
        ]
    )
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
