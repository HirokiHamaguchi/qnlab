"""PDF to PNG conversion utilities."""

from pathlib import Path

try:
    import fitz
except ImportError:
    print("Warning: PyMuPDF (fitz) not installed. PDF conversion will be skipped.")
    fitz = None


def convert_pdf_to_png(current_dir: Path) -> None:
    """Convert all PDF files in current_dir to PNG.

    Uses PyMuPDF to convert PDFs. Skips if fitz is not available.

    Args:
        current_dir: Directory containing PDF files to convert
    """
    if not fitz:
        return

    pdf_files = list(current_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        png_file = pdf_file.with_suffix(".png")
        if not png_file.exists():
            try:
                doc = fitz.open(str(pdf_file))
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                pix.save(str(png_file))
                doc.close()
                print(f"Converted {pdf_file.name} to {png_file.name}")
            except Exception as e:
                print(f"Error converting {pdf_file.name}: {e}")
