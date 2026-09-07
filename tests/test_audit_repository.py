"""Regression checks for the bounded repository release scan."""
import unittest
from src.audit_repository import scan


class AuditTests(unittest.TestCase):
    def test_public_university_url_is_not_a_local_path(self):
        self.assertEqual(scan(b"https://example.edu/home/researcher/paper.pdf"), [])

    def test_local_path_is_flagged(self):
        self.assertEqual(scan(b"/Users/" + b"example/book"), ["personal-local-path"])

    def test_synthetic_token_is_flagged(self):
        self.assertEqual(scan(b"ghp_" + b"a" * 36), ["github-token"])
