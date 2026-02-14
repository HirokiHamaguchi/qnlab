from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF

from qnlab.util.doc_paths import doc_imgs_dir, project_root

DEFAULT_DPI = 300
SELF_PATH = Path(__file__).resolve()
SCRIPT_ROOT = SELF_PATH.parent
PROJECT_ROOT = project_root()
DOC_IMGS_ROOT = doc_imgs_dir()


def iter_python_scripts() -> List[Path]:
    scripts = []
    for path in SCRIPT_ROOT.rglob("*.py"):
        if path == SELF_PATH or "__pycache__" in path.parts:
            continue
        if path.is_file():
            scripts.append(path)
    return sorted(scripts)


def run_scripts(script_paths: Iterable[Path]) -> None:
    for script in script_paths:
        rel = script.relative_to(PROJECT_ROOT)
        print(f"[script] uv run {rel}")
        subprocess.run(
            [
                "uv",
                "run",
                str(rel),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


def iter_pdfs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (path for path in root.rglob("*.pdf") if path.is_file())


def convert_pdf(pdf_path: Path) -> List[Path]:
    doc = fitz.open(pdf_path)
    multiple_pages = len(doc) > 1
    zoom = DEFAULT_DPI / 72  # fitz uses 72 DPI as base
    mat = fitz.Matrix(zoom, zoom)

    output_paths: List[Path] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)  # type: ignore
        suffix = f"_p{page_num + 1}" if multiple_pages else ""
        out_path = pdf_path.with_name(f"{pdf_path.stem}{suffix}.png")
        pix.save(out_path)
        output_paths.append(out_path)

    doc.close()
    return output_paths


def convert_all_pdfs() -> None:
    pdf_paths = sorted(iter_pdfs(DOC_IMGS_ROOT))
    if not pdf_paths:
        print("No PDF files found under doc/imgs.")
        return

    for pdf_path in pdf_paths:
        outputs = convert_pdf(pdf_path)
        for out_path in outputs:
            print(f"Generated {out_path}")


def main() -> None:
    DOC_IMGS_ROOT.mkdir(parents=True, exist_ok=True)
    script_paths = iter_python_scripts()
    run_scripts(script_paths)
    convert_all_pdfs()


if __name__ == "__main__":
    main()
