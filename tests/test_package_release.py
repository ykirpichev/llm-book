"""Release export must not expose files outside the declared source boundary."""
from pathlib import Path
import tempfile
import unittest

from src.package_release import TOP_LEVEL, TREES, checked_sources


class ReleasePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in TOP_LEVEL:
            (self.root / name).write_text("Public source\n")
        for name in TREES:
            (self.root / name).mkdir(parents=True)

    def test_export_excludes_git_credentials_and_local_outputs(self):
        for name in (".git/config", ".env", "output/demos/experiment.py",
                     ".local-archive/private.md", "docs/.private.md",
                     "examples/__pycache__/scratch.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("must stay local")
        (self.root / "manuscript/01_book.md").write_text("Chapter\n")
        self.assertEqual(set(checked_sources(self.root)), set(TOP_LEVEL) | {"manuscript/01_book.md"})

    def test_symlink_cannot_pull_in_external_content(self):
        (self.root / "docs/linked.md").symlink_to(self.root / "LICENSE")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            checked_sources(self.root)

    def test_working_tree_credentials_stop_export_without_printing_value(self):
        token = "ghp_" + "a" * 36
        (self.root / "docs/leak.md").write_text(token)
        with self.assertRaises(ValueError) as error:
            checked_sources(self.root)
        self.assertIn("github-token", str(error.exception))
        self.assertNotIn(token, str(error.exception))
