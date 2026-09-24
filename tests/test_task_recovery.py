from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from examples.task_recovery import TaskRunner


class TaskRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Path(self.temp.name)/'ledger.sqlite'
        self.effects = Path(self.temp.name)/'effects.sqlite'
        self.runner = TaskRunner(self.ledger, self.effects)
        self.addCleanup(self.runner.close)
        self.runner.prepare('task', 'a', 'key', 'body', generation=1, max_attempts=1)

    def resume(self, **kwargs):
        return self.runner.resume('task', authorized=True, current_generation=1, **kwargs)

    def test_checkpoint_gap_recovers_in_new_process_at_exhausted_budget(self):
        with self.assertRaises(RuntimeError):
            self.resume(fail_after_effect=True)
        self.assertEqual(self.runner.inspect('task')['status'], 'pending')
        code = """from examples.task_recovery import TaskRunner
import sys
r = TaskRunner(sys.argv[1],sys.argv[2])
try:
 print(r.resume('task',authorized=True,current_generation=1),r.inspect('task')['attempts'],r.effects.count())
finally:
 r.close()
"""
        result = subprocess.run([sys.executable,'-c',code,str(self.ledger),str(self.effects)],
                                cwd=Path(__file__).resolve().parents[1], check=True,
                                capture_output=True,text=True)
        self.assertEqual(result.stdout.strip(),'done 1 1')
        self.assertEqual(self.resume(), 'done')
        self.assertEqual(self.runner.inspect('task')['attempts'],1)

    def test_charged_attempt_without_receipt_does_not_reset(self):
        self.runner.db.execute("UPDATE tasks SET attempts=1 WHERE task_id='task'")
        self.assertEqual(self.resume(),'needs_reconciliation')
        self.runner.prepare('task','a','key','body',generation=1,max_attempts=1)
        self.assertEqual(self.resume(),'needs_reconciliation')
        self.assertEqual(self.runner.effects.count(),0)

    def test_stale_generation_cannot_dispatch_or_replay(self):
        self.assertEqual(self.runner.resume('task',authorized=True,current_generation=2),'stale')
        self.assertEqual(self.runner.effects.count(),0)
        self.resume()
        self.assertEqual(self.runner.resume('task',authorized=True,current_generation=2),'stale')
        self.assertEqual(self.runner.effects.count(),1)

    def test_revocation_blocks_completed_receipt(self):
        self.resume()
        with self.assertRaises(PermissionError):
            self.runner.resume('task',authorized=False,current_generation=1)

    def test_reprepare_cannot_change_payload_or_budget(self):
        for body,budget in [('other',1),('body',2)]:
            with self.assertRaises(ValueError):
                self.runner.prepare('task','a','key',body,generation=1,max_attempts=budget)
        self.assertEqual(self.runner.inspect('task')['attempts'],0)

    def test_conflicting_receipt_stops_recovery(self):
        self.runner.effects.save('a','key','other',authorized=True)
        with self.assertRaises(ValueError): self.resume()
        self.assertEqual(self.runner.inspect('task')['status'],'pending')

    def test_unknown_schema_and_missing_task_stop(self):
        self.runner.db.execute("UPDATE tasks SET schema_version=2")
        with self.assertRaises(ValueError): self.resume()
        with self.assertRaises(KeyError):
            self.runner.resume('missing',authorized=True,current_generation=1)

    def test_deleted_receipt_cannot_recreate_completed_effect(self):
        self.resume()
        self.runner.effects.db.execute('DELETE FROM receipts')
        self.assertEqual(self.resume(),'needs_reconciliation')
        self.assertEqual(self.runner.effects.count(),1)


if __name__ == '__main__':
    unittest.main()
