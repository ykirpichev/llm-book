#!/usr/bin/env python3
"""Package the public edition from an explicit source allowlist, without Git history.

Run through `make release` so the PDF has just been rebuilt and verified.
The bounded credential/path scan reports labels only, never matched values.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil
import zipfile

from pypdf import PdfReader
from src.audit_repository import scan

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = "engineering-large-language-models"
TOP_LEVEL = (
    ".gitignore", "LICENSE", "LICENSE-BOOK", "LICENSE-CODE", "README.md",
    "CONTRIBUTING.md", "Makefile", "requirements.txt",
)
TREES = {
    "manuscript": {".md"},
    "src": {".py", ".css", ".js"},
    "examples": {".py", ".cpp", ".md"},
    "tests": {".py"},
    "docs": {".md"},
    ".github/workflows": {".yml", ".yaml"},
}


def source_files(root: Path) -> list[Path]:
    files = [root / name for name in TOP_LEVEL]
    for name, suffixes in TREES.items():
        directory = root / name
        if (directory.resolve() != root.resolve() / name
                or not directory.is_dir()):
            raise ValueError(f"Missing or symlinked source directory: {name}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symlink in source tree: {path.relative_to(root)}")
            if path.is_file() and path.suffix in suffixes:
                if any(part.startswith(".") or part == "__pycache__"
                       for part in path.relative_to(directory).parts):
                    continue
                files.append(path)
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked source file: {path.relative_to(root)}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_sources(root: Path) -> dict[str, bytes]:
    sources = {}
    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        labels = scan(data)
        if labels:
            raise ValueError(f"Source scan requires review: {name}: {', '.join(labels)}")
        sources[name] = data
    return sources


def package(root: Path = ROOT) -> Path:
    sources = checked_sources(root)
    pdf_name = "engineering-large-language-models.pdf"
    pdf = root / "output" / "pdf" / pdf_name
    pdf_bytes = pdf.read_bytes()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    labels = scan(text.encode())
    if labels:
        raise ValueError(f"PDF scan requires review: {', '.join(labels)}")
    for term in ("Public Edition - September 2026", "CC BY 4.0", "MIT License"):
        if term not in text:
            raise ValueError(f"Rebuild the public edition PDF: missing {term}")

    destination = root / "output" / "release" / "public-2026-09-24-part7"
    destination.mkdir(parents=True, exist_ok=True)
    source_name = f"{SOURCE_ROOT}-source.zip"
    manifest = "".join(f"{sha256(data)}  {name}\n" for name, data in sources.items())
    entries = {**sources, "SOURCE-SHA256SUMS": manifest.encode()}
    with zipfile.ZipFile(destination / source_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(f"{SOURCE_ROOT}/{name}", date_time=(2026, 9, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    shutil.copyfile(pdf, destination / pdf_name)
    checksums = "".join(
        f"{sha256((destination / name).read_bytes())}  {name}\n"
        for name in (pdf_name, source_name)
    )
    (destination / "SHA256SUMS").write_text(checksums)
    print(f"PASS: {len(sources)} source files and extracted PDF text passed the bounded scan")
    print(f"Release assets: {destination}")
    return destination


if __name__ == "__main__":
    package()
