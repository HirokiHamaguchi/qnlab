"""Refresh existing performance-profile PNGs from the manuscript PDFs.

Run after regenerating the PDF profiles. Requires pdftoppm on PATH.
The README legend uses the joint-noise comparison's current legend.
"""

from pathlib import Path
import subprocess


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "doc" / "imgs" / "compare"
    pairs = [(image.with_suffix(".pdf"), image)
             for image in sorted(directory.glob("_pp_*.png"))]
    pairs.append((directory / "_legend_noise.pdf", directory / "_legend.png"))
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-r", "300", "-png",
             str(source), str(target.with_suffix(""))],
            check=True,
        )
        print(f"Updated: {target.name}")


if __name__ == "__main__":
    main()
