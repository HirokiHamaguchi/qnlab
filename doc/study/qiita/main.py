#!/usr/bin/env python3
"""
LaTeX to Markdown converter for study documents.
Processes all .tex files in the current directory and combines them into one Markdown file.
"""

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
    convert_pdf_to_png(current_dir.parent / "imgs" / "quasi_newton")
    convert_pdf_to_png(current_dir.parent / "imgs" / "modified_secant")

    all_markdown_lines = []

    tex_files = [f for f in sorted(current_dir.glob("[1-4]*.tex"))]
    assert tex_files, "No .tex files found in current directory."

    for tex_file in tex_files:
        print(f"Processing {tex_file.name}...")
        markdown_lines = process_latex_file(tex_file)

        all_markdown_lines.extend(
            [f"\n<!-- From {tex_file.name} -->\n", *markdown_lines, "\n"]
        )

    all_markdown_lines.extend(
        [
            "",
            "## 最後に",
            "",
            "以上です。お読みいただきありがとうございました。",
        ]
    )

    top_sentences = ["<!-- markdownlint-disable MD041 -->"]

    main_file = "0_main.tex"
    main_content = ""
    with open(current_dir / main_file, "r", encoding="utf-8") as f:
        main_content = f.read()
        main_content = main_content[
            main_content.rfind("\\else") + 6 : main_content.rfind("\\fi")
        ]
        main_content = "\n".join(line.strip() for line in main_content.splitlines())
        top_sentences.extend([f"\n<!-- From {main_file} -->\n", main_content])

    top_sentences.append(
        '\n<img width="100%" src="https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/doc/imgs/quasi_newton/sixhump.png?v=1" />\n'
    )

    top_sentences.extend(
        [
            "## 目次",
            "",
            "本記事はかなり網羅的かと思われます。興味のある箇所からお読みください。",
            "",
        ]
    )

    for line in all_markdown_lines:
        if line.startswith("#"):
            heading_level = line.count("#")
            heading_text = line[heading_level:].strip()
            escaped_heading_text = (
                heading_text.replace("(", "")
                .replace(")", "")
                .replace(" ", "-")
                .replace("–", "")
                .replace("--", "")
            )
            top_sentences.append(
                f"{'  ' * (heading_level - 2)}- [{heading_text}]({'#' + escaped_heading_text})"
            )

    all_markdown_lines = top_sentences + all_markdown_lines

    output_file = current_dir / "qiita" / "combined_output.md"
    output_content = "\n".join(all_markdown_lines)
    output_content = post_process_content(output_content)
    output_content = for_qiita_post_process(output_content)
    output_file.write_text(output_content, encoding="utf-8")
    print(f"\nConversion complete! Output written to {output_file}")
    print(f"Processed {len(tex_files)} files.")


if __name__ == "__main__":
    main()
