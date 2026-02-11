#!/usr/bin/env python3
"""
LaTeX to Markdown converter for study documents.
Processes all .tex files in the current directory and combines them into one Markdown file.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import fitz

# Constants
USE_GITHUB_URL = True
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


def find_matching(s: str, start_pos: int, bracket_type: str = "brace") -> int:
    r"""Find the position of the closing bracket matching the opening bracket at start_pos.

    Handles nested brackets correctly by counting bracket pairs.

    Args:
        s: The string to search
        start_pos: Position of the opening bracket
        bracket_type: Type of bracket to match - "brace" for {} or "bracket" for []

    Returns:
        Position of the matching closing bracket, or -1 if not found
    """
    bracket_pairs = {
        "brace": ("{", "}"),
        "bracket": ("[", "]"),
    }

    if bracket_type not in bracket_pairs:
        return -1

    open_char, close_char = bracket_pairs[bracket_type]

    if start_pos >= len(s) or s[start_pos] != open_char:
        return -1

    count = 1
    i = start_pos + 1

    while i < len(s) and count > 0:
        if s[i] == "\\":
            # Skip escaped characters
            i += 2
            continue
        elif s[i] == open_char:
            count += 1
        elif s[i] == close_char:
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
    end_brace = find_matching(block, start_brace, "brace")

    if end_brace == -1:
        return ""

    return block[start_brace + 1 : end_brace]


def preprocess_latex(content: str) -> str:
    r"""Preprocess LaTeX content.

    Removes \ifEn...\else...\fi blocks (keeps the \else part).
    Removes \ifSubfilesClassLoaded{...}{} blocks (ignoring whitespace).
    Keeps \label{...} commands for label tracking,
    they will be removed during environment block processing.
    """
    # Remove \ifEn...\else...\fi blocks (delete from \ifEn to \else, then delete \fi)
    # This removes the first branch and keeps the second branch
    pattern = r"^\s*\\ifEn\s*$.*?^\s*\\else\s*$"
    content = re.sub(pattern, "", content, flags=re.MULTILINE | re.DOTALL)
    # Remove the \fi command
    pattern = r"^\s*\\fi\s*$"
    content = re.sub(pattern, "", content, flags=re.MULTILINE)

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


def convert_nested_itemize_enumerate(content: str) -> str:
    """Convert nested itemize and enumerate environments within content.

    This function handles itemize/enumerate blocks that appear inside other
    environments (e.g., theorem, proposition).

    Args:
        content: The content string potentially containing itemize/enumerate blocks

    Returns:
        Content with itemize/enumerate blocks converted to Markdown
    """
    result = []
    i = 0
    lines = content.split("\n")

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for beginning of itemize or enumerate
        if stripped.startswith("\\begin{itemize}"):
            # Collect the entire itemize block
            block_lines = [line]
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if lines[i].strip().startswith("\\end{itemize}"):
                    i += 1
                    break
                i += 1
            # Convert and add
            converted = convert_itemize_to_md("\n".join(block_lines))
            if converted:
                result.append(converted.rstrip())
        elif stripped.startswith("\\begin{enumerate}"):
            # Collect the entire enumerate block
            block_lines = [line]
            i += 1
            while i < len(lines):
                block_lines.append(lines[i])
                if lines[i].strip().startswith("\\end{enumerate}"):
                    i += 1
                    break
                i += 1
            # Convert and add
            converted = convert_enumerate_to_md("\n".join(block_lines))
            if converted:
                result.append(converted.rstrip())
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def post_process_content(content: str) -> str:
    """Apply post-processing conversions to content.

    This is applied after main environment processing to handle conversions
    that should apply recursively inside other environments (like proof).

    Order matters: equation conversion first, then nested list environments,
    then cref conversion.
    """
    # First apply equation environment conversion
    content = convert_equation_environments(content)
    # Convert \href{url}{alt} to Markdown links
    content = convert_href_to_md(content)
    # Then convert nested itemize/enumerate
    content = convert_nested_itemize_enumerate(content)
    # Finally apply cref conversion
    content = convert_cref(content)
    return content


def convert_href_to_md(content: str) -> str:
    r"""Convert \href{url}{alt} to Markdown [alt](url)."""

    def href_replace(match: re.Match[str]) -> str:
        url = match.group(1)
        alt = match.group(2)
        return f"[{alt}]({url})"

    return re.sub(r"\\href\{([^}]+)\}\{([^}]+)\}", href_replace, content)


def convert_subfile_to_md(content: str) -> str:
    r"""Convert \subfile{filename.tex} to ![filename](filename.png)."""

    def subfile_replace(match: re.Match[str]) -> str:
        filepath = match.group(1)
        # Extract filename without extension
        filename = Path(filepath).stem
        # Build image path
        image_path = filename + ".png"
        if USE_GITHUB_URL:
            url = f"{GITHUB_RAW_URL_BASE}doc/{image_path}"
        else:
            url = image_path
        return f"![{filename}]({url})"

    return re.sub(r"\\subfile\{([^}]+)\}", subfile_replace, content)


def convert_pdf_to_png(current_dir: Path) -> None:
    pdf_files = list(current_dir.glob("999_*.pdf"))
    if not pdf_files:
        return

    print(f"Converting {len(pdf_files)} PDF files to PNG...")
    for pdf_path in pdf_files:
        png_path = pdf_path.with_suffix(".png")
        try:
            doc = fitz.open(str(pdf_path))
            # Render first page with good DPI
            pix = doc[0].get_pixmap(
                matrix=fitz.Matrix(5, 5)
            )  # 2x zoom for better quality
            pix.save(str(png_path))
            doc.close()
            print(f"  Converted: {pdf_path.name} -> {png_path.name}")
        except Exception as e:
            print(f"  Error converting {pdf_path.name}: {e}")


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
    image_entries = extract_figure_images(block)
    if not image_entries:
        return ""

    if len(image_entries) == 1:
        image_path = image_entries[0]["path"].replace(".pdf", ".png")
        if USE_GITHUB_URL:
            assert image_path.startswith("../")
            url = f"{GITHUB_RAW_URL_BASE}doc/{image_path.replace('../', '')}"
        else:
            url = image_path
        result = f"![{image_path}]({url})\n"
    else:
        img_tags = []
        for entry in image_entries:
            image_path = entry["path"].replace(".pdf", ".png")
            if USE_GITHUB_URL:
                assert image_path.startswith("../")
                url = f"{GITHUB_RAW_URL_BASE}doc/{image_path.replace('../', '')}"
            else:
                url = image_path
            width = entry.get("width_percent")
            if width is not None:
                img_tags.append(f'<img width="{width}%" src="{url}" />')
            else:
                img_tags.append(f'<img src="{url}" />')
        result = "".join(img_tags) + "\n"

    caption_text = extract_braced_content(block, "caption").replace("\n", " ").strip()
    if caption_text:
        print(f"Figure {counter} caption: {caption_text}")
        caption_text = post_process_content(caption_text)
        result += f"\n({ENV_DISPLAY_NAMES['figure']} {counter} {caption_text})\n"

    return post_process_content(result)


def extract_figure_images(block: str) -> List[Dict[str, str]]:
    r"""Extract image paths and widths from figure blocks.

    Supports simplified patterns with minipage widths like 0.5\textwidth or 0.33\textwidth.
    Returns a list of dicts with keys: path, width_percent (optional).
    """
    images: List[Dict[str, str]] = []

    # Find minipage blocks with includegraphics inside
    minipage_pattern = re.compile(
        r"\\begin\{minipage\}\{([0-9.]+)\\textwidth\}(.*?)\\end\{minipage\}",
        re.DOTALL,
    )
    include_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

    for match in minipage_pattern.finditer(block):
        width_str = match.group(1)
        inner = match.group(2)
        img_match = include_pattern.search(inner)
        if not img_match:
            continue
        path = img_match.group(1)
        try:
            width_percent = int(round(float(width_str) * 100))
        except ValueError:
            width_percent = None
        entry: Dict[str, str] = {"path": path}
        if width_percent is not None:
            entry["width_percent"] = str(width_percent)
        images.append(entry)

    # Fallback: if no minipage matches, collect all includegraphics
    if not images:
        for img_match in include_pattern.finditer(block):
            images.append({"path": img_match.group(1)})

    return images


def convert_table_to_md(block: str, counter: int) -> str:
    """Convert LaTeX table block to Markdown table.

    Handles tabular environments and converts them to Markdown table format.
    """
    # Extract tabular content
    tabular_match = re.search(
        r"\\begin\{tabular\}.*?\{([^}]+)\}(.*?)\\end\{tabular\}", block, re.DOTALL
    )
    assert tabular_match is not None, "No tabular environment found in table block."

    col_spec = tabular_match.group(1)
    tabular_content = tabular_match.group(2).replace("\\hline", "")

    # Parse rows (split by \\)
    rows = []
    for row_str in tabular_content.split("\\\\"):
        row_str = row_str.strip()
        if row_str:
            # Split cells by & and clean up
            cells = [cell.strip() for cell in row_str.split("&")]
            rows.append(cells)

    if rows:
        # Determine number of columns from column spec or first row
        num_cols = len(col_spec.replace("|", ""))

        # Build markdown table
        md_rows = []
        for i, row in enumerate(rows):
            # Pad row if necessary
            while len(row) < num_cols:
                row.append("")
            # Limit to num_cols
            row = row[:num_cols]
            md_rows.append("| " + " | ".join(row) + " |")

        # Add header separator after first row
        if md_rows:
            separator = "| " + " | ".join([":--:"] * num_cols) + " |"
            md_rows.insert(1, separator)

        table_content = "\n".join(md_rows) + "\n"
    else:
        table_content = ""

    caption_text = extract_braced_content(block, "caption").replace("\n", " ").strip()
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

    # Find the opening bracket after \begin{env_name}
    begin_pattern = r"\\begin\{" + re.escape(env_name) + r"\}"
    begin_match = re.search(begin_pattern, first_line)
    assert begin_match is not None, "Invalid environment block."
    start_pos = begin_match.end()
    if start_pos < len(first_line) and first_line[start_pos] == "[":
        # Find matching closing bracket, handling nested brackets
        end_pos = find_matching(first_line, start_pos, "bracket")
        bracket_content = first_line[start_pos + 1 : end_pos] if end_pos != -1 else ""
        if bracket_content:
            env_display = env_name.capitalize()
            header = f"**{env_display} {counter}** ({bracket_content})"
        else:
            env_display = env_name.capitalize()
            header = f"**{env_display} {counter}**"
    else:
        env_display = env_name.capitalize()
        header = f"**{env_display} {counter}**"

    content_lines = [
        line.strip()
        for i, line in enumerate(lines)
        if line.strip()
        and not (
            i == 0
            or (
                i == len(lines) - 1
                and line.strip().startswith("\\end{" + env_name + "}")
            )
        )
        and not line.strip().startswith(
            ("\\begin{" + env_name + "}", "\\end{" + env_name + "}")
        )
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
        # Convert \subfile{...} commands first
        line = convert_subfile_to_md(line)
        # Convert \href{url}{alt}
        line = convert_href_to_md(line)
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


def for_qiita_post_process(content: str) -> str:
    lines = content.splitlines()
    resList = []
    mathBlockOpen = False
    for line in lines:
        if line.strip() == "$$":
            resList.append("\n```math" if not mathBlockOpen else "```\n")
            mathBlockOpen = not mathBlockOpen
        elif line.strip() == "> $$":
            resList.append(">" if not mathBlockOpen else "> ```")
            resList.append("> ```math" if not mathBlockOpen else ">")
            mathBlockOpen = not mathBlockOpen
        else:
            resList.append(line)
    res = "\n".join(resList) + "\n"
    res = (
        res.replace("\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
    )
    res = res.replace("\\coloneqq", "\\mathrel{\\vcenter{:}}=").replace(
        "{dcases}", "{cases}"
    )
    for line in res.splitlines():
        if re.search(r"\\{[a-zA-Z0-9]", line):
            print(r"Warning: Add space after \{[a-zA-Z0-9] in line: " + line)
    res = res.replace("\\{", "\\lbrace").replace("\\}", "\\rbrace")
    if res.count("\\,"):
        print("Warning: \\, found. Use \\ instead.")

    cnt = res.count("\\|")
    if cnt % 2 == 1:
        raise ValueError(f"Odd number of \\| found: {cnt}")
    lastIdx = 0
    res2 = ""
    for vertNum, resIdx in enumerate(re.finditer(r"\\\|", res)):
        res2 += res[lastIdx : resIdx.start()]
        res2 += "\\lVert" if vertNum % 2 == 0 else "\\rVert"
        lastIdx = resIdx.end()
    res2 += res[lastIdx:]
    return res2


def main() -> None:
    """Main function to process all .tex files in current directory."""
    reset_global_counters()

    current_dir = Path(__file__).parent

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

    output_file = current_dir / "combined_output.md"
    output_content = "\n".join(all_markdown_lines)
    output_content = re.sub(r"\\label\{[^}]+\}", "", output_content)
    output_content = for_qiita_post_process(output_content)
    output_file.write_text(output_content, encoding="utf-8")
    print(f"\nConversion complete! Output written to {output_file}")
    print(f"Processed {len(tex_files)} files.")

    for filename, md_lines in each_markdown_lines:
        print(f"  {filename}: {len(md_lines)} lines")
        output_file = current_dir / f"{filename}_output.md"
        file_content = "\n".join(md_lines)
        file_content = re.sub(r"\\label\{[^}]+\}", "", file_content)
        file_content = for_qiita_post_process(file_content)
        output_file.write_text(file_content, encoding="utf-8")
        print(f"    Output written to {output_file}")


if __name__ == "__main__":
    main()
