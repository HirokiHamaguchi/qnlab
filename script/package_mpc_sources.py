"""Build and package only the manuscript dependencies for MPC submission.

Run with Python 3.10+ and latexmk/BibTeX on PATH. No Python packages are needed.
The response letter is submitted separately and is deliberately not included.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc"
MAIN = DOC / "main"
TARGETS = ("journal.tex", "journal_revised_MPC.tex")
SOURCE_SUFFIXES = {".tex", ".cls", ".clo", ".sty", ".bst", ".bib", ".bbl",
                   ".pdf", ".png", ".jpg", ".jpeg", ".eps"}


def compile_manuscripts(directory: Path) -> None:
    for target in TARGETS:
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", target],
            cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            print(result.stdout.decode("utf-8", errors="replace"))
            raise RuntimeError(f"Compilation failed: {directory / target}")
        log = (directory / Path(target).with_suffix(".log")).read_text(
            encoding="utf-8", errors="replace"
        )
        if "undefined" in log or "multiply defined" in log:
            raise RuntimeError(f"Unresolved or duplicate references: {target}")
        print(f"Compiled: {target}", flush=True)


def dependencies() -> list[Path]:
    files = {DOC / "L-BFGS.bib", MAIN / "spmpsci.bst"}
    for target in TARGETS:
        recorder = MAIN / Path(target).with_suffix(".fls")
        for line in recorder.read_text(encoding="utf-8").splitlines():
            if not line.startswith("INPUT "):
                continue
            path = Path(line[6:])
            if not path.is_absolute():
                path = MAIN / path
            path = path.resolve()
            # TeX Live packages are supplied by the recipient's TeX installation.
            if path.is_relative_to(DOC) and path.suffix.lower() in SOURCE_SUFFIXES:
                files.add(path)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(files, key=lambda path: path.relative_to(DOC).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DOC / "submission" / "mpc_source.zip",
        help="ZIP destination (default: doc/submission/mpc_source.zip)",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    compile_manuscripts(MAIN)
    files = dependencies()
    with tempfile.TemporaryDirectory(prefix="mpc_sources_", dir=output.parent) as temporary:
        staging = Path(temporary)
        contents = {}
        for source in files:
            name = "doc/" + source.relative_to(DOC).as_posix()
            contents[name] = source.read_bytes()
        contents["README.txt"] = (
            "MPC manuscript source files\n\n"
            "Requires a standard TeX Live installation with latexmk and BibTeX.\n"
            "Preserve the directory structure and compile from doc/main:\n"
            "  latexmk -pdf -interaction=nonstopmode -halt-on-error journal.tex\n"
            "  latexmk -pdf -interaction=nonstopmode -halt-on-error journal_revised_MPC.tex\n\n"
            "journal.tex produces the clean manuscript.\n"
            "journal_revised_MPC.tex shows substantive revisions in red.\n"
            "The .bib, .bst and generated .bbl files are included.\n"
            "PDF files under doc/imgs are figures, not compiled manuscripts.\n"
            "The response letter is submitted separately.\n"
        ).encode("utf-8")
        contents["MANIFEST.sha256"] = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in sorted(contents.items())
        ).encode("utf-8")
        archive = staging / "mpc_source.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
            for name, data in sorted(contents.items()):
                # Fixed metadata makes identical inputs produce identical archives.
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                package.writestr(info, data)
        # Verify the actual archive in isolation, without repository dependencies.
        extracted = staging / "verification"
        with zipfile.ZipFile(archive) as package:
            if package.testzip() is not None:
                raise RuntimeError("ZIP integrity check failed")
            package.extractall(extracted)
        compile_manuscripts(extracted / "doc" / "main")
        shutil.copyfile(archive, output)
    print(f"Created: {output}\nIncluded {len(files)} dependency files, README and manifest.")


if __name__ == "__main__":
    main()
