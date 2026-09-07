from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_book  # noqa: E402


class BuildBookTests(unittest.TestCase):
    def test_metadata_front_matter(self) -> None:
        meta, body = build_book.metadata_from_text(
            '---\ntitle: "Test Book"\nauthor: "Example"\n---\n\nBody\n'
        )
        self.assertEqual(meta.title, "Test Book")
        self.assertEqual(meta.author, "Example")
        self.assertEqual(body.strip(), "Body")

    def test_publication_metadata(self) -> None:
        meta, _ = build_book.metadata_from_text(
            '---\ncopyright_year: "2026"\npublication_date: "August 2026"\nkeywords: "LLM systems"\n---\n'
        )
        self.assertEqual(meta.copyright_year, "2026")
        self.assertEqual(meta.publication_date, "August 2026")
        self.assertEqual(meta.keywords, "LLM systems")

    def test_inline_markup_escapes_html(self) -> None:
        rendered = build_book.inline_markup("A < B and **bold** with `code`")
        self.assertIn("A &lt; B", rendered)
        self.assertIn("<b>bold</b>", rendered)
        self.assertIn('<font name="Courier">code</font>', rendered)

    def test_table_parser_discards_separator_row(self) -> None:
        rows = build_book.parse_table(
            ["| A | B |", "| --- | --- |", "| one | two |"]
        )
        self.assertEqual(rows, [["A", "B"], ["one", "two"]])

    def test_slug_is_stable_and_bounded(self) -> None:
        first = build_book.slugify("A Heading With Punctuation!")
        second = build_book.slugify("A Heading With Punctuation!")
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 60)


if __name__ == "__main__":
    unittest.main()
