#!/usr/bin/env python3
"""Structural and text checks for the generated handbook PDF."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


REQUIRED_TERMS = [
    "Speculative Decoding",
    "KL Divergence",
    "Prefill",
    "Tiled Matrix Multiplication",
    "Continuous Batching",
    "Technical Leadership",
    "Recruiter-Derived Interview Masterclass",
]


def verify(path: Path) -> None:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    if page_count < 60:
        raise SystemExit(f"FAIL: expected a substantial handbook, found {page_count} pages")

    meta = reader.metadata
    if not meta or "Principal ML Systems" not in (meta.title or ""):
        raise SystemExit("FAIL: missing or incorrect PDF title metadata")

    samples = sorted({0, 1, 2, page_count // 4, page_count // 2, 3 * page_count // 4, page_count - 1})
    extracted = []
    with pdfplumber.open(str(path)) as pdf:
        for index in samples:
            page = pdf.pages[index]
            text = page.extract_text() or ""
            extracted.append(text)
            if index > 0 and len(text.strip()) < 20:
                raise SystemExit(f"FAIL: suspiciously empty page {index + 1}")
    corpus = "\n".join((page.extract_text() or "") for page in reader.pages)
    for term in REQUIRED_TERMS:
        if term.lower() not in corpus.lower():
            raise SystemExit(f"FAIL: required topic missing: {term}")

    bad_tokens = ["turn0", "turn1", "PLACEHOLDER", "TODO", "�"]
    for token in bad_tokens:
        if token in corpus:
            raise SystemExit(f"FAIL: leaked placeholder or extraction defect: {token}")

    # Catch accidental extreme repetition while allowing headers and footers.
    normalized = [re.sub(r"\s+", " ", line).strip() for line in corpus.splitlines()]
    normalized = [line for line in normalized if len(line) > 45]
    if normalized:
        most_common = max(normalized.count(line) for line in set(normalized))
        if most_common > max(12, page_count // 3):
            raise SystemExit("FAIL: suspicious repeated long line across the document")

    result = subprocess.run(["pdfinfo", str(path)], check=True, text=True, capture_output=True)
    if "Page size:" not in result.stdout:
        raise SystemExit("FAIL: pdfinfo could not read page geometry")

    print(f"PASS: {path}")
    print(f"pages={page_count}")
    print(f"bytes={path.stat().st_size}")
    print("sample_pages=" + ",".join(str(i + 1) for i in samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    verify(args.pdf)


if __name__ == "__main__":
    main()
