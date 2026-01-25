from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF

DOC_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = DOC_ROOT / "png"
DEFAULT_DPI = 300


def iter_pdfs(root: Path) -> Iterable[Path]:
    for pdf_path in root.rglob("*.pdf"):
        if pdf_path.is_file() and not pdf_path.is_relative_to(OUTPUT_DIR):
            yield pdf_path


def build_prefix(pdf_path: Path) -> str:
    rel_parts = pdf_path.relative_to(DOC_ROOT).with_suffix("").parts
    return "_".join(rel_parts)


def convert_pdf(pdf_path: Path) -> List[Path]:
    doc = fitz.open(pdf_path)
    prefix = build_prefix(pdf_path)
    multiple_pages = len(doc) > 1
    zoom = DEFAULT_DPI / 72  # fitz uses 72 DPI as base
    mat = fitz.Matrix(zoom, zoom)

    output_paths: List[Path] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        suffix = f"_p{page_num + 1}" if multiple_pages else ""
        out_path = OUTPUT_DIR / f"{prefix}{suffix}.png"
        pix.save(out_path)
        output_paths.append(out_path)

    doc.close()
    return output_paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(iter_pdfs(DOC_ROOT))
    if not pdf_paths:
        print("No PDF files found under doc/imgs.")
        return

    for pdf_path in pdf_paths:
        outputs = convert_pdf(pdf_path)
        for out_path in outputs:
            print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
