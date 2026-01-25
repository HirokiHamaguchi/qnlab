from __future__ import annotations

from pathlib import Path

__all__ = [
    "project_root",
    "doc_imgs_root",
    "doc_imgs_dir",
    "script_imgs_root",
]

_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _FILE.parents[2]
_DOC_IMGS_ROOT = _PROJECT_ROOT / "doc" / "imgs"
_SCRIPT_IMGS_ROOT = _PROJECT_ROOT / "script" / "imgs"


def project_root() -> Path:
    """Return the repository root that hosts the qnlab package."""
    return _PROJECT_ROOT


def doc_imgs_root(create: bool = True) -> Path:
    """Return the root directory for documentation images."""
    if create:
        _DOC_IMGS_ROOT.mkdir(parents=True, exist_ok=True)
    return _DOC_IMGS_ROOT


def doc_imgs_dir(*parts: str, create: bool = True) -> Path:
    """Return (and optionally create) a path inside doc/imgs."""
    target = doc_imgs_root(create=create).joinpath(*parts)
    if create:
        target.mkdir(parents=True, exist_ok=True)
    return target


def script_imgs_root(create: bool = False) -> Path:
    """Return the root directory that stores the image scripts."""
    if create:
        _SCRIPT_IMGS_ROOT.mkdir(parents=True, exist_ok=True)
    return _SCRIPT_IMGS_ROOT
