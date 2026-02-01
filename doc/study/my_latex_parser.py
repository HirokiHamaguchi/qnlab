#!/usr/bin/env python3
"""
LaTeX to Markdown converter for study documents.
Processes all .tex files in the current directory and combines them into one Markdown file.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Constants
USE_GITHUB_URL = False
GITHUB_RAW_URL_BASE = "https://raw.githubusercontent.com/HirokiHamaguchi/qnlab/master/"

# Environment types
MATH_ENVS = [
    "theorem",
    "proposition",
    "lemma",
    "definition",
    "corollary",
    "remark",
    "example",
    "assumption",
]
EQUATION_ENVS = ["equation", "equation*", "align", "align*"]
OTHER_ENVS = ["proof"]

# Display names for environments
ENV_DISPLAY_NAMES = {
    "figure": "Fig.",
    "table": "Table",
    "theorem": "Theorem",
    "proposition": "Proposition",
    "lemma": "Lemma",
    "definition": "Definition",
    "corollary": "Corollary",
    "remark": "Remark",
    "example": "Example",
    "assumption": "Assumption",
}

# Global counters and label mapping across all files
GLOBAL_COUNTERS: Dict[str, int] = {
    "figure": 0,
    "table": 0,
    "theorem": 0,
    "proposition": 0,
    "lemma": 0,
    "definition": 0,
    "corollary": 0,
    "remark": 0,
    "example": 0,
    "assumption": 0,
}

# Map label names to their type and number
LABEL_MAP: Dict[str, Tuple[str, int]] = {}


def reset_global_counters() -> None:
    """Reset global counters and label map."""
    global GLOBAL_COUNTERS, LABEL_MAP
    for key in GLOBAL_COUNTERS:
        GLOBAL_COUNTERS[key] = 0
    LABEL_MAP.clear()


def find_matching_brace(s: str, start_pos: int) -> int:
    r"""Find the position of the closing brace matching the opening brace at start_pos.

    Handles nested braces correctly by counting brace pairs.

    Args:
        s: The string to search
        start_pos: Position of the opening brace {

    Returns:
        Position of the matching closing brace, or -1 if not found
    """
    if start_pos >= len(s) or s[start_pos] != "{":
        return -1

    count = 1
    i = start_pos + 1

    while i < len(s) and count > 0:
        if s[i] == "\\":
            # Skip escaped characters
            i += 2
            continue
        elif s[i] == "{":
            count += 1
        elif s[i] == "}":
            count -= 1
        i += 1

    if count == 0:
        return i - 1
    return -1


def extract_braced_content(block: str, command: str) -> str:
    r"""Extract content from a LaTeX command with braced argument.

    Handles nested braces correctly.

    Args:
        block: The block content
        command: The command name (e.g., 'caption', 'begin')

    Returns:
        Content within the braces, or empty string if not found
    """
    pattern = rf"\\{command}\{{"
    match = re.search(pattern, block)
    if not match:
        return ""

    start_brace = match.end() - 1  # Position of the opening brace
    end_brace = find_matching_brace(block, start_brace)

    if end_brace == -1:
        return ""

    return block[start_brace + 1 : end_brace]


def preprocess_latex(content: str) -> str:
    r"""Preprocess LaTeX content.

    Removes \ifSubfilesClassLoaded{...}{} blocks (ignoring whitespace).
    Keeps \label{...} commands for label tracking,
    they will be removed during environment block processing.
    """
    # Remove \ifSubfilesClassLoaded{...}{} blocks
    pattern = r"\\ifSubfilesClassLoaded\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\{\}"
    return re.sub(pattern, "", content, flags=re.DOTALL)


def convert_cref(line: str) -> str:
    r"""Convert \cref{label} to appropriate reference format."""

    def cref_replace(match: re.Match[str]) -> str:
        label = match.group(1)

        if label not in LABEL_MAP:
            return match.group(0)

        label_type, label_num = LABEL_MAP[label]
        display_name = ENV_DISPLAY_NAMES.get(label_type)

        if display_name:
            return f"{display_name} {label_num}"
        return match.group(0)

    return re.sub(r"\\cref\{([^}]+)\}", cref_replace, line)


def convert_section_commands(line: str) -> str:
    """Convert LaTeX section commands to Markdown headers."""
    section_mappings = [
        ("section", "##"),
        ("subsection", "###"),
        ("subsubsection", "####"),
        ("paragraph", "#####"),
        ("subparagraph", "######"),
    ]
    for cmd, heading in section_mappings:
        pattern = rf"\\{cmd}\{{([^}}]*)\}}"
        replacement = rf"{heading} \1"
        line = re.sub(pattern, replacement, line)
    return line


def post_process_content(content: str) -> str:
    """Apply post-processing conversions to content.

    This is applied after main environment processing to handle conversions
    that should apply recursively inside other environments (like proof).

    Order matters: equation conversion first, then cref conversion.
    """
    # First apply equation environment conversion
    content = convert_equation_environments(content)
    # Then apply cref conversion
    content = convert_cref(content)
    return content


def convert_equation_environments(block: str) -> str:
    r"""Convert LaTeX equation environments to Markdown $$ format.

    Converts \begin{equation}...\end{equation} and similar environments
    to $$\n...\n$$ format for Markdown rendering.

    Handles: equation, equation*, align, align*
    """
    lines = block.split("\n")
    result_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for begin of equation environments
        if re.match(r"\\begin\{(equation\*?|align\*?)\}", stripped):
            # Add newline and $$ before the environment
            result_lines.append("")
            result_lines.append("$$")
            result_lines.append(line)
        # Check for end of equation environments
        elif re.match(r"\\end\{(equation\*?|align\*?)\}", stripped):
            # Add $$ and newline after the environment
            result_lines.append(line)
            result_lines.append("$$")
            result_lines.append("")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def convert_figure_to_md(block: str, counter: int) -> str:
    r"""Convert LaTeX figure block to Markdown.

    Converts \includegraphics paths to GitHub raw content URLs and PDF to PNG.
    Extracts caption and adds it below the figure.
    """
    match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
    if not match:
        return ""

    image_path = match.group(1).replace(".pdf", ".png")

    if USE_GITHUB_URL:
        assert image_path.startswith("../")
        url = f"{GITHUB_RAW_URL_BASE}{image_path.replace('../', '')}"
    else:
        url = image_path

    result = f"![{image_path}]({url})\n"

    caption_text = extract_braced_content(block, "caption")
    if caption_text:
        print(f"Figure {counter} caption: {caption_text}")
        caption_text = post_process_content(caption_text)
        result += f"\n({ENV_DISPLAY_NAMES['figure']} {counter} {caption_text})\n"

    return post_process_content(result)


def convert_table_to_md(block: str, counter: int) -> str:
    """Convert LaTeX table block to Markdown table."""
    lines = [
        line.strip()
        for line in block.split("\n")
        if line.strip() and not line.strip().startswith(("\\begin", "\\end"))
    ]

    table_content = "\n".join(lines) + "\n" if lines else ""

    caption_text = extract_braced_content(block, "caption")
    if caption_text:
        caption_text = post_process_content(caption_text)
        table_content += f"\n({ENV_DISPLAY_NAMES['table']} {counter} {caption_text})\n"

    return post_process_content(table_content)


def convert_itemize_to_md(block: str) -> str:
    """Convert LaTeX itemize block to Markdown list."""
    result = []
    for line in block.split("\n"):
        line = line.strip()
        if line.startswith("\\item"):
            content = post_process_content(line[5:].strip())
            result.append(f"- {content}")
    return "\n".join(result) + "\n" if result else ""


def convert_enumerate_to_md(block: str) -> str:
    """Convert LaTeX enumerate block to Markdown numbered list."""
    result = []
    counter = 1
    for line in block.split("\n"):
        line = line.strip()
        if line.startswith("\\item"):
            content = post_process_content(line[5:].strip())
            result.append(f"{counter}. {content}")
            counter += 1
    return "\n".join(result) + "\n" if result else ""


def convert_proof_to_md(block: str) -> str:
    r"""Convert LaTeX proof environment to HTML details element.

    \begin{proof}..\end{proof} -> <details><summary>Proof</summary>...\n</details>
    """
    lines = block.split("\n")
    content_lines = [
        line.strip()
        for i, line in enumerate(lines)
        if line.strip()
        and not (
            i == 0 or (i == len(lines) - 1 and line.strip().startswith("\\end{proof}"))
        )
        and not line.strip().startswith(("\\begin{proof}", "\\end{proof}"))
    ]

    content = post_process_content("\n".join(content_lines))

    if content:
        return f"<details>\n<summary>Proof</summary>\n\n{content}\n\n</details>\n"
    return "<details>\n<summary>Proof</summary>\n</details>\n"


def convert_math_env_to_md(block: str, env_name: str, counter: int) -> str:
    """Convert LaTeX theorem-like environments to Markdown.

    Args:
        block: The entire environment block
        env_name: Name of the environment (theorem, proposition, etc.)
        counter: Current counter for this environment type

    Returns:
        Markdown formatted string
    """
    lines = block.split("\n")
    if not lines:
        return ""

    first_line = lines[0].strip()
    optional_match = re.search(
        r"\\begin\{" + re.escape(env_name) + r"\}\[([^\]]+)\]", first_line
    )

    env_display = env_name.capitalize()
    header = (
        f"**{env_display} {counter}** ({optional_match.group(1)})"
        if optional_match
        else f"**{env_display} {counter}**"
    )

    content_lines = [
        line.strip()
        for i, line in enumerate(lines)
        if line.strip()
        and not (i == 0 or (i == len(lines) - 1 and line.strip().startswith("\\end")))
        and not line.strip().startswith(("\\begin", "\\end"))
    ]

    content = post_process_content("\n".join(content_lines))

    return f"\n\n{header}\n\n{content}\n" if content else f"\n\n{header}\n"


class LatexToMarkdownConverter:
    """LaTeX to Markdown converter class."""

    def __init__(self, filepath: Path):
        """Initialize converter with filepath."""
        self.filepath = filepath
        self.lines: List[str] = []
        self.markdown_lines: List[str] = []
        self.i = 0
        # Map from (env_name, line_number) to counter value
        self.env_counters: Dict[Tuple[str, int], int] = {}

    def process_environment_block(self, line: str) -> bool:
        """Process environment blocks (figure, table, itemize, enumerate, theorem-like, proof).

        Returns:
            True if the block was processed, False otherwise.
        """
        all_envs = self._get_all_envs_pattern()
        environment_match = re.match(r"\\begin\{(" + all_envs + r")\}", line)
        if not environment_match:
            return False

        env_name = environment_match.group(1)
        # Record the starting line number for this environment
        env_start_line = self.i
        block = self._collect_environment_block(env_name, line)
        cleaned_block, labels = self.extract_and_remove_labels(block)

        converted = self._convert_environment(
            env_name, cleaned_block, labels, env_start_line
        )

        if converted:
            self.markdown_lines.append(converted)

        return True

    def _get_all_envs_pattern(self) -> str:
        """Get regex pattern for all supported environments."""
        return (
            "figure|table|itemize|enumerate|"
            + "|".join(MATH_ENVS)
            + "|"
            + "|".join(EQUATION_ENVS).replace("*", r"\*")
            + "|"
            + "|".join(OTHER_ENVS)
        )

    def _collect_environment_block(self, env_name: str, first_line: str) -> str:
        """Collect lines until \\end{env_name} is found."""
        block_lines = [first_line]
        self.i += 1

        while self.i < len(self.lines):
            block_line = self.lines[self.i].strip()
            block_lines.append(block_line)

            if f"\\end{{{env_name}}}" in block_line:
                self.i += 1
                return "\n".join(block_lines)
            self.i += 1

        raise ValueError(f"Missing \\end{{{env_name}}} in {self.filepath}")

    def _convert_environment(
        self, env_name: str, block: str, labels: List[str], env_start_line: int
    ) -> str:
        """Convert environment block to Markdown based on environment type."""
        if env_name in ("figure", "table"):
            return self._convert_numbered_env(env_name, block, labels, env_start_line)
        elif env_name == "itemize":
            return convert_itemize_to_md(block)
        elif env_name == "enumerate":
            return convert_enumerate_to_md(block)
        elif env_name in MATH_ENVS:
            return self._convert_numbered_env(env_name, block, labels, env_start_line)
        elif env_name in EQUATION_ENVS:
            return convert_equation_environments(block)
        elif env_name == "proof":
            return convert_proof_to_md(block)
        return ""

    def _convert_numbered_env(
        self, env_name: str, block: str, labels: List[str], env_start_line: int
    ) -> str:
        """Convert numbered environment (figure, table, math) to Markdown."""
        # Get the counter value that was assigned during preregister_labels
        counter = self.env_counters.get(
            (env_name, env_start_line), GLOBAL_COUNTERS[env_name]
        )

        if env_name == "figure":
            return convert_figure_to_md(block, counter)
        elif env_name == "table":
            return convert_table_to_md(block, counter)
        else:
            return convert_math_env_to_md(block, env_name, counter)

    def should_skip_line(self, line: str) -> bool:
        """Check if a line should be skipped (metadata commands, etc.)."""
        skip_commands = [
            "\\title",
            "\\author",
            "\\date",
            "\\maketitle",
            "\\orcidlink",
        ]
        for cmd in skip_commands:
            if line.startswith(cmd):
                return True
        return False

    def preregister_labels(self) -> None:
        """Pre-scan the file to register all labels before processing.

        This ensures that \\cref{label} references can be converted correctly.
        """
        in_document = False
        i = 0
        all_envs = self._get_all_envs_pattern()

        while i < len(self.lines):
            line = self.lines[i]

            if not in_document:
                if r"\begin{document}" in line:
                    in_document = True
                i += 1
                continue

            if r"\end{document}" in line:
                break

            environment_match = re.match(
                r"\\begin\{(" + all_envs + r")\}", line.strip()
            )

            if environment_match:
                env_name = environment_match.group(1)
                i = self._preregister_env_labels(env_name, i)
            else:
                i += 1

    def _preregister_env_labels(self, env_name: str, start_idx: int) -> int:
        """Pre-register labels for a single environment block."""
        block_lines = [self.lines[start_idx].strip()]
        i = start_idx + 1

        while i < len(self.lines):
            block_line = self.lines[i].strip()
            block_lines.append(block_line)

            if f"\\end{{{env_name}}}" in block_line:
                i += 1
                break
            i += 1

        block = "\n".join(block_lines)
        _, labels = self.extract_and_remove_labels(block)

        if env_name in ("figure", "table") or env_name in MATH_ENVS:
            GLOBAL_COUNTERS[env_name] += 1
            counter = GLOBAL_COUNTERS[env_name]
            # Store the counter value for this specific environment instance
            self.env_counters[(env_name, start_idx)] = counter
            self.register_labels(labels, env_name, counter)

        return i

    def extract_and_remove_labels(self, block: str) -> Tuple[str, List[str]]:
        r"""Extract labels from block and return cleaned block + list of labels.

        Returns:
            (cleaned_block, label_list): cleaned block with labels removed, and list of label names
        """
        labels = [match.group(1) for match in re.finditer(r"\\label\{([^}]+)\}", block)]
        # Remove \label{...} along with preceding newline and whitespace if present
        cleaned_block = re.sub(r"\n\s*\\label\{[^}]+\}", "", block)
        # Also remove \label{...} without preceding newline (fallback)
        cleaned_block = re.sub(r"\\label\{[^}]+\}", "", cleaned_block)
        return cleaned_block, labels

    def register_labels(self, labels: List[str], env_name: str, counter: int) -> None:
        """Register labels in LABEL_MAP."""
        for label in labels:
            LABEL_MAP[label] = (env_name, counter)

    def process_normal_line(self, line: str) -> None:
        """Process a normal (non-environment) line."""
        # Skip metadata commands
        if self.should_skip_line(line):
            self.i += 1
            return

        # Convert \cref references
        line = convert_cref(line)

        # Convert section commands
        line = convert_section_commands(line)
        self.markdown_lines.append(line)
        self.i += 1

    def process_file(self) -> List[str]:
        """Process the LaTeX file and return list of Markdown lines."""
        content = self.filepath.read_text(encoding="utf-8")
        content = preprocess_latex(content)

        self.lines = content.split("\n")
        self.markdown_lines = []

        self.preregister_labels()

        in_document = False
        self.i = 0

        while self.i < len(self.lines):
            line = self.lines[self.i]

            if not in_document:
                if r"\begin{document}" in line:
                    in_document = True
                self.i += 1
                continue

            if r"\end{document}" in line:
                break

            line = line.strip()

            if line.startswith("%"):
                self.i += 1
                continue

            if self.process_environment_block(line):
                continue

            self.process_normal_line(line)

        return self.markdown_lines


def process_latex_file(filepath: Path) -> List[str]:
    """Process a single LaTeX file and return list of Markdown lines."""
    converter = LatexToMarkdownConverter(filepath)
    return converter.process_file()


def main() -> None:
    """Main function to process all .tex files in current directory."""
    reset_global_counters()

    current_dir = Path(__file__).parent
    exclude_files = {"0_en.tex"}
    tex_files = [
        f for f in sorted(current_dir.glob("*.tex")) if f.name not in exclude_files
    ]

    if not tex_files:
        print("No .tex files found in current directory.")
        return

    all_markdown_lines = ["# Study of Quasi-Newton Methods\n"]

    for tex_file in tex_files:
        print(f"Processing {tex_file.name}...")
        markdown_lines = process_latex_file(tex_file)

        if markdown_lines:
            all_markdown_lines.extend(
                [f"\n<!-- From {tex_file.name} -->\n", *markdown_lines, "\n"]
            )

    output_file = current_dir / "combined_output.md"
    output_content = "\n".join(all_markdown_lines)
    output_content = re.sub(r"\\label\{[^}]+\}", "", output_content)

    output_file.write_text(output_content, encoding="utf-8")

    print(f"\nConversion complete! Output written to {output_file}")
    print(f"Processed {len(tex_files)} files.")


if __name__ == "__main__":
    main()
