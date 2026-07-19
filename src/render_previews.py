#!/usr/bin/env python3
"""Render selected PDF pages to PNG previews without system font-cache state."""

from __future__ import annotations

import argparse
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("pages", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("output/previews"))
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(args.pdf))
    scale = args.dpi / 72.0
    for number in args.pages:
        if number < 1 or number > len(document):
            raise SystemExit(f"page {number} outside 1..{len(document)}")
        page = document[number - 1]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        destination = args.output_dir / f"page-{number}.png"
        image.save(destination)
        print(destination)


if __name__ == "__main__":
    main()
