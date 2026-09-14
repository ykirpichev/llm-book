#!/usr/bin/env python3
"""Render every authored figure with its real caption into a compact QA PDF.

This proof is temporary. The final deliverable remains the complete book.
"""
import argparse
import re
from pathlib import Path

from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Spacer
from build_book import ROOT, STYLES, diagram, inline_markup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/pdfs/visual-pass/figure-proofs.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    story = []
    seen = set()
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        for name, caption in re.findall(r"^:::diagram ([a-z_]+)\|(.+)$", path.read_text(), re.M):
            if name in seen:
                continue
            seen.add(name)
            story.extend([
                Paragraph(inline_markup(name), STYLES["small"]), Spacer(1, 5),
                diagram(name, 407),
                Paragraph(inline_markup(caption), STYLES["caption"]), PageBreak(),
            ])
            print(f"{len(seen)} {name}")
    SimpleDocTemplate(str(args.output), pagesize=(447, 455),
                      leftMargin=20, rightMargin=20, topMargin=12, bottomMargin=12).build(story)
    print(args.output)


if __name__ == "__main__":
    main()
