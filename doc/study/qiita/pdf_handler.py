"""PDF to PNG conversion utilities."""

from pathlib import Path

import pymupdf  # type: ignore[import-untyped]


def convert_pdf_to_png(current_dir: Path) -> None:
    """Convert all PDF files in current_dir to PNG.

    Uses PyMuPDF to render PDF pages as images.

    Args:
        current_dir: Directory containing PDF files to convert
    """

    pdf_files = list(current_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to convert.")
    for pdf_file in pdf_files:
        if pdf_file.parent.name == "main" and any(
            pdf_file.stem.startswith(str(i)) for i in range(1, 5)
        ):
            continue
        if "sixhump" in pdf_file.name:
            continue

        png_file = pdf_file.with_suffix(".png")
        try:
            with pymupdf.open(str(pdf_file)) as doc:
                page = doc[0]
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                pix.save(str(png_file))
            print(f"Converted {pdf_file.name} to {png_file.name}")
        except (IndexError, OSError, RuntimeError, ValueError) as e:
            print(f"Error converting {pdf_file.name}: {e}")
