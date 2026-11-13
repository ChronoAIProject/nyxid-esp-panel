"""Fail-closed model-input schedule tests, no model or fresh material."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from hey_batch_schedule_v1 import BatchGuard,batch_digest


class GuardChecks(unittest.TestCase):
    def setup_guard(self,root,rows=None):
        values=(np.ones((2,3,4),dtype=np.float32),np.array([[1],[0]],dtype=np.float32),np.array([1,20],dtype=np.float32))
        if rows is None:
            rows=[dict(index=i,kind='train' if i<=2 else 'ptq',tensor_sha256=batch_digest(*values)) for i in range(1,4)]
        schedule=root/'expected.jsonl';schedule.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return BatchGuard(schedule,root,train_steps=2,ptq_batches=1),values

    def test_complete_and_extra(self):
        with tempfile.TemporaryDirectory() as t:
            guard,values=self.setup_guard(Path(t))
            for _ in range(3):
                returned=guard(*values)
                self.assertTrue(all(a is b for a,b in zip(returned,values)))
            self.assertEqual(guard.finish()['verified_batches'],3)
            with self.assertRaises(ValueError):guard(*values)

    def test_changed_input_label_or_weight(self):
        for which in range(3):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as t:
                guard,values=self.setup_guard(Path(t))
                changed=[v.copy() for v in values];changed[which].flat[0]+=1
                try:
                    with self.assertRaises(ValueError):guard(*changed)
                    self.assertEqual(guard.count,0)
                    row=json.loads((Path(t)/'actual-batch-schedule.jsonl').read_text())
                    self.assertFalse(row['matches_expected'])
                finally:
                    if guard.log:guard.log.close()

    def test_incomplete(self):
        with tempfile.TemporaryDirectory() as t:
            guard,values=self.setup_guard(Path(t));guard(*values)
            try:
                with self.assertRaises(ValueError):guard.finish()
            finally:guard.log.close()

    def test_wrong_phase_or_order(self):
        for index,kind in [(2,'train'),(1,'ptq')]:
            with self.subTest(index=index,kind=kind),tempfile.TemporaryDirectory() as t:
                rows=[dict(index=index,kind=kind,tensor_sha256='0'*64)]*3
                with self.assertRaises(ValueError):self.setup_guard(Path(t),rows)


if __name__=='__main__':unittest.main()
