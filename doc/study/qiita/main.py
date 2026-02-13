#!/usr/bin/env python3
"""
LaTeX to Markdown converter for study documents.
Processes all .tex files in the current directory and combines them into one Markdown file.
"""

import re
import sys
from pathlib import Path

# Ensure imports work from doc/study directory
sys.path.insert(0, str(Path(__file__).parent))

from pdf_handler import convert_pdf_to_png
from postprocessing import for_qiita_post_process, post_process_content
from processor import get_global_state, process_latex_file


def main() -> None:
    """Main function to process all .tex files in current directory."""
    global_state = get_global_state()
    global_state.reset()

    current_dir = Path(__file__).parent.parent.resolve()

    # Convert PDFs to PNGs
    convert_pdf_to_png(current_dir)
    main_file = "0_main.tex"
    tex_files = [
        f for f in sorted(current_dir.glob("[1-4]*.tex")) if f.name != main_file
    ]

    assert tex_files, "No .tex files found in current directory."

    all_markdown_lines = ["<!-- markdownlint-disable MD041 -->"]
    each_markdown_lines = []

    for tex_file in tex_files:
        print(f"Processing {tex_file.name}...")
        markdown_lines = process_latex_file(tex_file)

        all_markdown_lines.extend(
            [f"\n<!-- From {tex_file.name} -->\n", *markdown_lines, "\n"]
        )
        each_markdown_lines.append((tex_file.name, markdown_lines))

    output_file = current_dir / "qiita" / "combined_output.md"
    output_content = "\n".join(all_markdown_lines)
    output_content = post_process_content(output_content)
    output_content = for_qiita_post_process(output_content)
    output_file.write_text(output_content, encoding="utf-8")
    print(f"\nConversion complete! Output written to {output_file}")
    print(f"Processed {len(tex_files)} files.")


if __name__ == "__main__":
    main()
