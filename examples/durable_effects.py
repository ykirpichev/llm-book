"""Local atomic effect/receipt fixture; not authentication or a remote workflow engine."""
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory


class DraftStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS receipts (
                tenant TEXT NOT NULL, operation_key TEXT NOT NULL,
                body TEXT NOT NULL, draft_id INTEGER NOT NULL REFERENCES drafts(id),
                PRIMARY KEY (tenant, operation_key));
        """)

    def close(self):
        self.db.close()

    def save(self, tenant, key, body, *, authorized, failpoint=None):
        """The trusted host supplies identity and a fresh authorization decision.

        Failpoints model rollback before commit and loss of the reply after commit.
        Both tables are local to one SQLite database. No remote effect is protected.
        """
        if not all(isinstance(v, str) and v for v in (tenant, key, body)):
            raise ValueError("tenant, key, and body must be nonempty strings")
        if authorized is not True:
            raise PermissionError("current authorization required, including replay")
        if failpoint not in (None, "before_commit", "after_commit"):
            raise ValueError("unknown failpoint")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            receipt = self.db.execute(
                "SELECT body, draft_id FROM receipts WHERE tenant=? AND operation_key=?",
                (tenant, key)).fetchone()
            if receipt is not None:
                if receipt[0] != body:
                    raise ValueError("operation key reused with a different payload")
                draft_id = receipt[1]
            else:
                draft_id = self.db.execute(
                    "INSERT INTO drafts(tenant, body) VALUES (?, ?)",
                    (tenant, body)).lastrowid
                self.db.execute("INSERT INTO receipts VALUES (?, ?, ?, ?)",
                                (tenant, key, body, draft_id))
            if failpoint == "before_commit":
                raise RuntimeError("interrupted before commit")
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        if failpoint == "after_commit":
            raise TimeoutError("reply lost after commit")
        return draft_id

    def count(self):
        """Administrative test inspection, never an exposed user endpoint."""
        return self.db.execute("SELECT count(*) FROM drafts").fetchone()[0]


def demo():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "drafts.sqlite"
        store = DraftStore(path)
        try:
            store.save("tenant-a", "draft-1", "Example", authorized=True,
                       failpoint="after_commit")
        except TimeoutError:
            pass
        store.close()
        store = DraftStore(path)
        try:
            first = store.save("tenant-a", "draft-1", "Example", authorized=True)
            second = store.save("tenant-a", "draft-1", "Example", authorized=True)
            print(f"same_receipt={first == second} drafts={store.count()}")
        finally:
            store.close()


if __name__ == "__main__":
    demo()
