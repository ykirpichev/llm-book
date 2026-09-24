import concurrent.futures
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.durable_effects import DraftStore


class DurableEffectsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "drafts.sqlite"
        self.store = DraftStore(self.path)
        self.addCleanup(self.store.close)

    def save(self, **kwargs):
        return self.store.save("a", "key", "body", authorized=True, **kwargs)

    def test_lost_reply_reopens_in_new_process(self):
        with self.assertRaises(TimeoutError):
            self.save(failpoint="after_commit")
        expected = self.save()
        code = """from examples.durable_effects import DraftStore
import sys
s = DraftStore(sys.argv[1])
try:
    print(s.save('a', 'key', 'body', authorized=True), s.count())
finally:
    s.close()
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.path)],
                                cwd=Path(__file__).resolve().parents[1],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), f"{expected} 1")

    def test_before_commit_rolls_back_both_tables(self):
        with self.assertRaises(RuntimeError):
            self.save(failpoint="before_commit")
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM receipts").fetchone()[0], 0)
        self.save()
        self.assertEqual(self.store.count(), 1)

    def test_hard_exit_with_uncommitted_effect(self):
        code = """import os, sqlite3, sys
s = sqlite3.connect(sys.argv[1], isolation_level=None)
s.execute('BEGIN IMMEDIATE')
s.execute("INSERT INTO drafts(tenant, body) VALUES ('a', 'body')")
os._exit(23)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.path)])
        self.assertEqual(result.returncode, 23)
        self.assertEqual(self.store.count(), 0)
        self.save()
        self.assertEqual(self.store.count(), 1)

    def test_payload_conflict_preserves_original(self):
        first = self.save()
        with self.assertRaises(ValueError):
            self.store.save("a", "key", "changed", authorized=True)
        self.assertEqual(self.save(), first)
        self.assertEqual(self.store.count(), 1)

    def test_revocation_prevents_receipt_replay(self):
        self.save()
        with self.assertRaises(PermissionError):
            self.store.save("a", "key", "body", authorized=False)
        self.assertEqual(self.store.count(), 1)

    def test_tenant_keys_are_independent(self):
        first = self.save()
        second = self.store.save("b", "key", "body", authorized=True)
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.count(), 2)

    def test_duplicate_concurrent_connections(self):
        def attempt(_):
            store = DraftStore(self.path)
            try:
                return store.save("a", "key", "body", authorized=True)
            finally:
                store.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(attempt, range(8)))
        self.assertEqual(len(set(receipts)), 1)
        self.assertEqual(self.store.count(), 1)

    def test_invalid_requests_do_not_write(self):
        for tenant, key, body in [("", "k", "b"), ("a", "", "b"), ("a", "k", [] )]:
            with self.assertRaises(ValueError):
                self.store.save(tenant, key, body, authorized=True)
        with self.assertRaises(ValueError):
            self.save(failpoint="typo")
        with self.assertRaises(PermissionError):
            self.store.save("a", "k", "b", authorized="yes")
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
