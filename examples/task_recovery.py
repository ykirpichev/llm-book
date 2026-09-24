"""Single-runner task recovery across separate local ledger/effect databases.

The trusted host supplies authority and generation. Effect attempts are bounded;
model/token/deadline budgets, remote services and concurrent task runners are absent.
"""
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from examples.durable_effects import DraftStore


class TaskRunner:
    def __init__(self, ledger_path, effect_path):
        self.db = sqlite3.connect(ledger_path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, operation_key TEXT NOT NULL,
            body TEXT NOT NULL, generation INTEGER NOT NULL, schema_version INTEGER NOT NULL,
            max_attempts INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending', receipt INTEGER)""")
        self.effects = DraftStore(effect_path)

    def close(self):
        self.effects.close()
        self.db.close()

    def prepare(self, task_id, tenant, key, body, *, generation, max_attempts):
        if not all(isinstance(x, str) and x for x in (task_id, tenant, key, body)):
            raise ValueError("nonempty string identities and payload required")
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("positive integer attempt limit required")
        values = (task_id, tenant, key, body, generation, 1, max_attempts)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is not None:
                if tuple(row)[:7] != values:
                    raise ValueError("existing intent cannot be redefined")
            else:
                self.db.execute("""INSERT INTO tasks
                    (task_id,tenant,operation_key,body,generation,schema_version,max_attempts)
                    VALUES (?,?,?,?,?,?,?)""", values)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def resume(self, task_id, *, authorized, current_generation, fail_after_effect=False):
        # In this fixture, authority and generation are stable during one resume.
        if authorized is not True:
            raise PermissionError("current host authorization required")
        if type(current_generation) is not int or current_generation < 0:
            raise ValueError("invalid current generation")
        row = self.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        if row['schema_version'] != 1:
            raise ValueError("unsupported task schema")
        if row['generation'] != current_generation:
            return 'stale'
        # Read-only reconciliation does not consume another EFFECT attempt.
        receipt = self.effects.db.execute(
            "SELECT body,draft_id FROM receipts WHERE tenant=? AND operation_key=?",
            (row['tenant'], row['operation_key'])).fetchone()
        if receipt is not None:
            if receipt[0] != row['body']:
                raise ValueError("receipt payload conflicts with durable intent")
            draft_id = receipt[1]
        else:
            if row['status'] == 'done' or row['attempts'] >= row['max_attempts']:
                return 'needs_reconciliation'
            # Autocommit reserves the attempt before touching the effect service.
            self.db.execute("UPDATE tasks SET attempts=attempts+1 WHERE task_id=?", (task_id,))
            draft_id = self.effects.save(row['tenant'], row['operation_key'], row['body'],
                                         authorized=True)
            if fail_after_effect:
                raise RuntimeError("process interrupted before task completion checkpoint")
        self.db.execute("UPDATE tasks SET status='done',receipt=? WHERE task_id=?",
                        (draft_id, task_id))
        return 'done'

    def inspect(self, task_id):
        """Administrative test helper, never a user-facing authorization bypass."""
        return dict(self.db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone())


def demo():
    with TemporaryDirectory() as directory:
        ledger, effects = (Path(directory)/n for n in ('ledger.sqlite', 'effects.sqlite'))
        runner = TaskRunner(ledger, effects)
        runner.prepare('task', 'tenant', 'draft-1', 'Example', generation=1, max_attempts=1)
        try:
            runner.resume('task', authorized=True, current_generation=1, fail_after_effect=True)
        except RuntimeError:
            pass
        runner.close()
        runner = TaskRunner(ledger, effects)
        try:
            status = runner.resume('task', authorized=True, current_generation=1)
            print(f"status={status} attempts={runner.inspect('task')['attempts']} drafts={runner.effects.count()}")
        finally:
            runner.close()


if __name__ == '__main__':
    demo()
