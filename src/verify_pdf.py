#!/usr/bin/env python3
"""Structural and text checks for the generated handbook PDF."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


REQUIRED_TERMS = [
    "Speculative Decoding",
    "KL Divergence",
    "Prefill",
    "Hierarchical Matrix Multiplication",
    "Continuous Batching",
    "Technical Leadership",
    "Cross-Layer Design Prompt Bank",
    "Recent State of the Art",
    "FlashAttention-3",
    "AgentDojo",
]


def flatten_outline(items: Iterable[object]) -> list[str]:
    """Return the text titles from pypdf's nested outline representation."""
    titles: list[str] = []
    for item in items:
        if isinstance(item, list):
            titles.extend(flatten_outline(item))
            continue
        title = getattr(item, "title", None)
        if title:
            titles.append(str(title))
    return titles


def verify(path: Path) -> None:
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    if page_count < 60:
        raise SystemExit(f"FAIL: expected a substantial handbook, found {page_count} pages")

    meta = reader.metadata
    if not meta or "Engineering Large Language Models" not in (meta.title or ""):
        raise SystemExit("FAIL: missing or incorrect PDF title metadata")
    if meta.author != "Yury Kirpichev":
        raise SystemExit("FAIL: missing or incorrect PDF author metadata")
    if not meta.keywords or "large language models" not in meta.keywords.lower():
        raise SystemExit("FAIL: missing PDF keywords")

    catalog = reader.trailer["/Root"]
    if str(catalog.get("/Lang", "")) != "en-US":
        raise SystemExit("FAIL: missing or incorrect PDF document language")

    expected_width, expected_height = 7 * 72, 10 * 72
    for index, page in enumerate(reader.pages):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - expected_width) > 0.5 or abs(height - expected_height) > 0.5:
            raise SystemExit(f"FAIL: unexpected geometry on page {index + 1}: {width}x{height}")

    outline_titles = flatten_outline(reader.outline)
    if len(outline_titles) < 45:
        raise SystemExit(f"FAIL: suspiciously short PDF outline: {len(outline_titles)} entries")
    if len(outline_titles) != len(set(outline_titles)):
        raise SystemExit("FAIL: duplicate titles in PDF outline")

    external_links: list[str] = []
    for page in reader.pages:
        for annotation_ref in page.get("/Annots", []):
            annotation = annotation_ref.get_object()
            action = annotation.get("/A")
            if action and action.get("/URI"):
                external_links.append(str(action["/URI"]))
    insecure_links = [uri for uri in external_links if not uri.startswith("https://")]
    if insecure_links:
        raise SystemExit(f"FAIL: non-HTTPS external links: {insecure_links[:3]}")

    samples = sorted({0, 1, 2, page_count // 4, page_count // 2, 3 * page_count // 4, page_count - 1})
    extracted = []
    sparse_pages: list[tuple[int, int]] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if index > 0 and len(text.strip()) < 40:
                raise SystemExit(f"FAIL: suspiciously empty page {index + 1}")
            if index > 0 and len(text.strip()) < 180:
                sparse_pages.append((index + 1, len(text.strip())))
        for index in samples:
            page = pdf.pages[index]
            text = page.extract_text() or ""
            extracted.append(text)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    corpus = "\n".join(page_texts)
    # This book ends with a short closing section. A continuation page can
    # exceed the sparse-character threshold while still being an orphan.
    closing_pages = [i for i, text in enumerate(page_texts) if "Final principle" in text]
    if closing_pages != [page_count - 1]:
        raise SystemExit("FAIL: closing section is missing or split; inspect final pages")
    for term in REQUIRED_TERMS:
        if term.lower() not in corpus.lower():
            raise SystemExit(f"FAIL: required topic missing: {term}")
    for publication_term in ["Copyright © 2026 Yury Kirpichev", "All rights reserved", "Working Draft - September 2026"]:
        if publication_term.lower() not in corpus.lower():
            raise SystemExit(f"FAIL: publication front matter missing: {publication_term}")

    bad_tokens = [
        "turn0",
        "turn1",
        "PLACEHOLDER",
        "TODO",
        "�",
        "■",
        "□",
    ]
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
    print(f"outline_entries={len(outline_titles)}")
    print(f"external_links={len(external_links)}")
    print("sparse_pages=" + (",".join(f"{page}:{chars}" for page, chars in sparse_pages) or "none"))
    print("sample_pages=" + ",".join(str(i + 1) for i in samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()
    verify(args.pdf)


if __name__ == "__main__":
    main()
