#!/usr/bin/env python3
"""Render every PDF page into labeled contact sheets for visual QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/pdfs"))
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--thumbnail-width", type=int, default=180)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(args.pdf))
    first_page = document[0]
    source_w, source_h = first_page.get_size()
    thumb_w = args.thumbnail_width
    thumb_h = int(round(thumb_w * source_h / source_w))
    label_h = 20
    gutter = 10
    cell_w = thumb_w + gutter
    cell_h = thumb_h + label_h + gutter
    per_sheet = args.columns * args.rows
    sheets = math.ceil(len(document) / per_sheet)
    font = ImageFont.load_default()

    for sheet_index in range(sheets):
        canvas = Image.new(
            "RGB",
            (args.columns * cell_w + gutter, args.rows * cell_h + gutter),
            "#d8dee4",
        )
        draw = ImageDraw.Draw(canvas)
        start = sheet_index * per_sheet
        end = min(start + per_sheet, len(document))
        for page_index in range(start, end):
            local = page_index - start
            col = local % args.columns
            row = local // args.columns
            x = gutter + col * cell_w
            y = gutter + row * cell_h
            page = document[page_index]
            bitmap = page.render(scale=thumb_w / source_w)
            image = bitmap.to_pil().convert("RGB")
            canvas.paste(image, (x, y))
            label = f"page {page_index + 1}"
            draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill="#ffffff")
            draw.text((x + 5, y + thumb_h + 4), label, fill="#152238", font=font)
        destination = args.output_dir / f"contact-{sheet_index + 1:02d}.png"
        canvas.save(destination)
        print(destination)


if __name__ == "__main__":
    main()
