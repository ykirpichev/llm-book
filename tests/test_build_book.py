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
        names = re.findall(r'name == "([a-z_]+)"', source) + list(build_book.FIGURES)
        for name in names:
            drawing = build_book.diagram(name, 407)
            x0, y0, x1, y1 = drawing.getBounds()
            self.assertGreaterEqual(x0, -1, name)
            self.assertGreaterEqual(y0, -1, name)
            self.assertLessEqual(x1, drawing.width + 1, name)
            self.assertLessEqual(y1, drawing.height + 1, name)

    def test_manuscript_diagram_names_resolve(self) -> None:
        import re
        for path in sorted((ROOT / "manuscript").glob("*.md")):
            for name in re.findall(r"^:::diagram ([a-z_]+)\|", path.read_text(), re.M):
                with self.subTest(name=name):
                    build_book.diagram(name, 407)
        with self.assertRaises(ValueError):
            build_book.diagram("missing_figure", 407)

    def figure_labels(self, name):
        # Wording guards supplement, but do not replace, semantic visual review.
        return [item.text for item in build_book.diagram(name, 407).contents
                if isinstance(item, build_book.String)]

    def test_request_diagram_uses_client_timestamps(self):
        labels = self.figure_labels("request_lifecycle")
        self.assertIn("client receipt", labels)
        self.assertIn("t1: token 1", labels)
        self.assertIn("TTFT = t1 - t0", labels)

    def test_decoder_diagram_separates_sums_from_outputs(self):
        labels = self.figure_labels("decoder_block")
        self.assertEqual(labels.count("+"), 2)
        self.assertIn("X next", labels)
        self.assertFalse(any(label.startswith("Add:") for label in labels))

    def test_speculation_diagram_names_both_branches(self):
        labels = " ".join(self.figure_labels("speculative_decoding"))
        self.assertIn("correction on rejection", labels)
        self.assertIn("target bonus if all proposals pass", labels)
        self.assertIn("EOS", labels)

    def test_pipeline_diagram_has_four_balanced_microbatches(self):
        labels = self.figure_labels("pipeline_bubbles")
        for batch in "ABCD":
            self.assertEqual(labels.count(batch), 4)
        self.assertTrue(any("3/7" in label for label in labels))
        self.assertTrue(any("Backward is not shown" in label for label in labels))

    def test_agent_diagram_routes_trusted_authority(self):
        labels = self.figure_labels("agent_trust_boundary")
        self.assertIn("Trusted caller authority bypasses the model", labels)
        self.assertIn("Policy gate", labels)

    def test_figure_and_caption_share_keep_group(self) -> None:
        from unittest.mock import Mock
        source = Mock()
        source.read_text.return_value = ":::diagram token_alignment|A caption.\n"
        _, story = build_book.build_story([source], 407)
        group = story[-1]
        self.assertIsInstance(group, build_book.KeepTogether)
        self.assertIsInstance(group._content[1], build_book.Drawing)
        self.assertEqual(group._content[2].getPlainText(), "A caption.")

    def test_figure_keeps_its_introducing_heading(self) -> None:
        from unittest.mock import Mock
        source = Mock()
        source.read_text.return_value = "### Supervision\n\n:::diagram token_alignment|A caption.\n"
        _, story = build_book.build_story([source], 407)
        group = story[-1]
        self.assertIsInstance(group, build_book.KeepTogether)
        self.assertIsInstance(group._content[0], build_book.Heading)
        self.assertIsInstance(group._content[2], build_book.Drawing)

    def test_equation_keeps_its_explanation(self) -> None:
        from unittest.mock import Mock
        source = Mock()
        source.read_text.return_value = ":::equation O = A V|Weighted values.\n"
        _, story = build_book.build_story([source], 407)
        self.assertIsInstance(story[-1], build_book.KeepTogether)
        self.assertEqual(story[-1]._content[-1].getPlainText(), "Weighted values.")


if __name__ == "__main__":
    unittest.main()
