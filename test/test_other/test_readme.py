"""Test to verify that submodules/README.md table of contents covers all subdirectories."""

import re
from pathlib import Path


def test_submodules_readme_toc():
    """Test that all submodule directories are listed in the README.md table of contents."""
    # Get the project root directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    submodules_dir = project_root / "submodules"
    readme_path = submodules_dir / "README.md"

    # Get all directories in submodules/ (excluding hidden directories and README.md)
    actual_dirs = sorted(
        [
            d.name
            for d in submodules_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    )

    # Read README.md
    readme_content = readme_path.read_text()

    # Extract table of contents entries (lines starting with "  - [")
    # Pattern: lines like "  - [DirL-BFGS](#dirl-bfgs)"
    toc_pattern = r"^\s*-\s*\[([^\]]+)\]\(#[^\)]+\)$"
    toc_entries = []

    for line in readme_content.split("\n"):
        match = re.match(toc_pattern, line)
        if match:
            entry = match.group(1)
            # Skip the main "submodules" entry
            if entry != "submodules":
                toc_entries.append(entry)

    # Normalize directory names to match TOC entries
    # TOC uses escaped underscores and may have different formatting
    def normalize_for_toc(name):
        """Normalize directory name to match TOC entry format."""
        # Replace underscores with escaped underscores as they appear in markdown
        return name.replace("_", r"\_")

    # Create a mapping of TOC entries to actual directory names
    toc_to_dir = {}
    for entry in toc_entries:
        # Remove escape characters for comparison
        normalized_entry = entry.replace(r"\_", "_")
        toc_to_dir[normalized_entry] = entry

    # Find directories in TOC (normalized)
    dirs_in_toc = sorted(toc_to_dir.keys())

    # Find missing directories (in filesystem but not in TOC)
    missing_in_toc = set(actual_dirs) - set(dirs_in_toc)

    # Find extra entries (in TOC but not in filesystem)
    extra_in_toc = set(dirs_in_toc) - set(actual_dirs)

    # Print diagnostic information
    if missing_in_toc or extra_in_toc:
        print("\n=== Diagnostic Information ===")
        print("\nActual directories in submodules/:")
        for d in actual_dirs:
            print(f"  - {d}")

        print("\nDirectories listed in TOC:")
        for d in dirs_in_toc:
            print(f"  - {d}")

        if missing_in_toc:
            print("\nMissing in TOC (exist in filesystem but not in README):")
            for d in sorted(missing_in_toc):
                print(f"  - {d}")

        if extra_in_toc:
            print("\nExtra in TOC (in README but not in filesystem):")
            for d in sorted(extra_in_toc):
                print(f"  - {d}")

    # Assert that all directories are covered
    assert not missing_in_toc, (
        f"The following directories are missing from the README.md table of contents: "
        f"{sorted(missing_in_toc)}"
    )

    assert not extra_in_toc, (
        f"The following entries in the README.md table of contents do not have "
        f"corresponding directories: {sorted(extra_in_toc)}"
    )

    print(
        f"\n✓ All {len(actual_dirs)} subdirectories are properly documented in README.md"
    )


if __name__ == "__main__":
    test_submodules_readme_toc()
