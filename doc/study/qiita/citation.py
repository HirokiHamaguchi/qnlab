"""Citation handling for LaTeX to Markdown conversion."""

import json
import re
from pathlib import Path
from typing import Dict

try:
    from config import CITE_MAPPING_FILE
except ImportError:
    # Fallback for relative imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import CITE_MAPPING_FILE


def load_cite_mapping(json_file: str = CITE_MAPPING_FILE) -> Dict[str, str]:
    """Load citation key mappings from JSON file.

    Args:
        json_file: Path to the JSON file containing citation mappings

    Returns:
        Dictionary mapping citation keys to their bibliography information
    """
    json_path = Path(__file__).parent / json_file
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Data is a flat dictionary: {citationKey: "bibliography info"}
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {json_file}: {e}")
        return {}


def convert_citep(content: str) -> str:
    r"""Convert \citep{key1,key2,...} to Markdown footnote references.

    Loads citation mappings from cite_mapping.json and replaces \citep commands
    with Markdown footnote references [^key]. Multiple keys can be separated by commas.

    Also appends footnote definitions to the end of content in the format:
    [^key]: bibliography information

    If a key is not found in the mapping, it defaults to "TODO".

    Args:
        content: The content string containing \citep commands

    Returns:
        Content with \citep commands converted to Markdown footnote references,
        with footnote definitions appended at the end
    """
    cite_mapping = load_cite_mapping()
    used_keys = set()

    def citep_replace(match: re.Match[str]) -> str:
        # Extract optional arguments and citation keys
        opt1 = match.group(1)  # First optional argument
        opt2 = match.group(2)  # Second optional argument
        cite_keys = match.group(3)  # Citation keys

        # Split by comma and strip whitespace
        keys = [key.strip() for key in cite_keys.split(",")]

        # Convert each key and track used keys
        footnote_refs = []
        for key in keys:
            used_keys.add(key)
            footnote_refs.append(f"[^{key}]")

        result = "".join(footnote_refs)

        # Add optional arguments if present
        optional_parts = []
        if opt1:
            optional_parts.append(opt1)
        if opt2:
            optional_parts.append(opt2)

        if optional_parts:
            result += f" ({', '.join(optional_parts)})"

        return result

    # Replace all \citep commands (with optional arguments)
    content_converted = re.sub(
        r"\\citep(?:\[([^\]]*)\])?(?:\[([^\]]*)\])?\{([^}]+)\}",
        citep_replace,
        content
    )

    # Append footnote definitions at the end
    if used_keys:
        footnote_definitions = []
        for key in sorted(used_keys):
            # Get bibliography info from mapping, default to "TODO" if not found
            bib_info = cite_mapping.get(key, "TODO")
            footnote_definitions.append(f"[^{key}]: {bib_info}")

        # Add blank line before footnotes if content doesn't end with newline
        if content_converted and not content_converted.endswith("\n"):
            content_converted += "\n"

        content_converted += "\n" + "\n".join(footnote_definitions)

    return content_converted
