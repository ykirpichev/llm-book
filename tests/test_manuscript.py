"""Editorial checks that can fail before a PDF is built."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ManuscriptTests(unittest.TestCase):
    def test_decision_index_names_existing_chapters_and_parts(self):
        part_names = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX")
        destinations = set()
        for number, part in enumerate(part_names, 1):
            paths = list((ROOT / "manuscript").glob(f"{number:02d}_*.md"))
            self.assertEqual(len(paths), 1)
            for line in paths[0].read_text().splitlines():
                if line.startswith("## "):
                    destinations.add(f"{part} — {line[3:]}")
        source = (ROOT / "manuscript" / "09_appendices.md").read_text()
        index = source.split("### Decision index\n", 1)[1].split("### ", 1)[0]
        rows = [line for line in index.splitlines()
                if line.startswith("| ")
                and not line.startswith(("| If the symptom", "| ---"))]
        self.assertEqual(len(rows), 17, "Decision index entries changed")
        for row in rows:
            for destination in row.split("|")[-2].strip().split("; "):
                self.assertIn(destination, destinations)

    def test_chapter_coverage_ledger_is_complete(self):
        ledger = (ROOT / "docs" / "coverage-audit-2026-09.md").read_text()
        chapters = [line[3:] for path in sorted((ROOT / "manuscript").glob("*.md"))
                    for line in path.read_text().splitlines() if line.startswith("## ")]
        self.assertEqual(len(chapters), len(set(chapters)), "Duplicate chapter title")
        for chapter in chapters:
            self.assertEqual(ledger.count(f"| {chapter} |"), 1, chapter)

    def test_display_equations_have_one_caption_separator(self):
        for path in sorted((ROOT / "manuscript").glob("*.md")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if line.startswith(":::equation "):
                    self.assertEqual(line.count("|"), 1, f"{path.name}:{number}")

    def test_five_pass_review_covers_current_chapters(self):
        ledger = (ROOT / "docs" / "five-pass-review-2026-09-13.md").read_text()
        for path in sorted((ROOT / "manuscript").glob("*.md")):
            for line in path.read_text().splitlines():
                if line.startswith("## "):
                    self.assertEqual(ledger.count(f"| {line[3:]} |"), 1, line)

    def test_code_blocks_have_status_and_are_closed(self):
        for path in sorted((ROOT / "manuscript").glob("*.md")):
            inside = False
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if line.startswith("```"):
                    if not inside:
                        self.assertIn("Example status:", "\n".join(lines[max(0, i-3):i]),
                                      f"Unlabeled snippet in {path.name}:{i+1}")
                    inside = not inside
            self.assertFalse(inside, f"Unclosed code block in {path.name}")

    def test_explicitly_runnable_python_blocks(self):
        executed = 0
        for path in sorted((ROOT / "manuscript").glob("*.md")):
            lines = path.read_text().splitlines()
            i = 0
            while i < len(lines):
                if lines[i] == "```python" and "Example status: Runnable" in "\n".join(lines[max(0, i-3):i]):
                    end = lines.index("```", i + 1)
                    code = "\n".join(lines[i+1:end])
                    exec(compile(code, str(path), "exec"), {"__name__": "__example__"})
                    executed += 1
                    i = end
                i += 1
        self.assertGreater(executed, 0, "No runnable manuscript examples tested")
