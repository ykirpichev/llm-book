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

    def test_table_heading_shares_its_keep_group(self) -> None:
        from unittest.mock import Mock
        source = Mock()
        source.read_text.return_value = "### Three masks\n\n| A | B |\n| --- | --- |\n| one | two |\n"
        _, story = build_book.build_story([source], 400)
        group = story[-1]
        self.assertIsInstance(group, build_book.KeepTogether)
        self.assertIsInstance(group._content[0], build_book.Heading)
        self.assertIsInstance(group._content[1], build_book.Table)

    def test_inline_code_preserves_multiplication_and_emphasis(self) -> None:
        rendered = build_book.inline_markup("`B * H` and `S * d` with **bold**")
        self.assertIn('<font name="Courier">B * H</font>', rendered)
        self.assertIn('<font name="Courier">S * d</font>', rendered)
        self.assertIn("<b>bold</b>", rendered)
        self.assertNotIn("<i>", rendered)
        self.assertNotIn("\x00", rendered)

    def test_slug_is_stable_and_bounded(self) -> None:
        first = build_book.slugify("A Heading With Punctuation!")
        second = build_book.slugify("A Heading With Punctuation!")
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 60)

    def test_code_wrap_preserves_all_characters(self) -> None:
        source = "result = " + "abcdefghij" * 30
        panel = build_book.CodePanel(source, "python", 200)
        panel.wrap(200, 1000)
        self.assertEqual("".join(line for line, _ in panel.lines), source)
        self.assertTrue(panel.lines[1][1])
        parts = panel.split(200, 100)
        self.assertEqual(len(parts), 2)
        for part in parts:
            part.wrap(200, 1000)
        self.assertEqual([line for part in parts for line in part.lines], panel.lines)

    def test_chapter_band_owns_navigation_anchor(self) -> None:
        band = build_book.ChapterBand("Chapter 1", "A chapter", "01")
        self.assertEqual(band.anchor, build_book.slugify("A chapter"))
        self.assertEqual(band.level, 2)
        self.assertTrue(band.toc)

    def test_diagram_content_stays_inside_canvas(self) -> None:
        import re
        source = (ROOT / "src" / "build_book.py").read_text()
        for name in re.findall(r'name == "([a-z_]+)"', source):
            drawing = build_book.diagram(name, 407)
            x0, y0, x1, y1 = drawing.getBounds()
            self.assertGreaterEqual(x0, -1, name)
            self.assertGreaterEqual(y0, -1, name)
            self.assertLessEqual(x1, drawing.width + 1, name)
            self.assertLessEqual(y1, drawing.height + 1, name)


if __name__ == "__main__":
    unittest.main()
