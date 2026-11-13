"""Analytic admission tests only; never synthesize or score acceptance audio."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import prepare_mike_acceptance_v1 as admission


def decision():
    metrics=dict(macro=.9,standalone=.9,command=.9,pre_source_positive_events=0,
                 negatives=dict(common=dict(activated_clips=0),cross_target=dict(activated_clips=0)))
    grid=[dict(threshold=t,synthetic=copy.deepcopy(metrics),background_upper95=.4,eligible=True)
          for t in admission.GRID]
    return dict(minimum_macro=.8,grid=grid,selected=copy.deepcopy(grid[-1]),
                threshold=grid[-1]['threshold'],viable_operating_point=True,
                candidate_promoted=False,acceptance_authorized=False)


class Admission(unittest.TestCase):
    def test_original_grid_and_highest_threshold_tie(self):
        self.assertEqual(admission.validate_selection(decision(),.8)['threshold'],admission.GRID[-1])

    def test_null_failed_stale_and_noncanonical_decisions_rejected(self):
        for field,value in [('selected',None),('viable_operating_point',False),
                            ('threshold',.5),('minimum_macro',.7),('acceptance_authorized',True)]:
            with self.subTest(field=field):
                d=decision();d[field]=value
                with self.assertRaises(AssertionError):admission.validate_selection(d,.8)
        d=decision();d['grid'].pop()
        with self.assertRaises(AssertionError):admission.validate_selection(d,.8)

    def test_forged_eligibility_rejected(self):
        d=decision();d['grid'][0]['synthetic']['command']=.5
        with self.assertRaises(AssertionError):admission.validate_selection(d,.8)

    def test_actual_unfinished_development_cannot_write_lock_or_initialize_engine(self):
        self.assertFalse(admission.DEV_PINS.exists())
        self.assertFalse(admission.LOCK.exists())
        self.assertFalse(admission.OUT.exists())
        with patch.dict(sys.modules,{'onnxruntime':None}):
            with self.assertRaisesRegex(RuntimeError,'missing development prerequisite'):admission.write_lock()
            with self.assertRaisesRegex(RuntimeError,'no development-passed committed lock'):admission.generate()
        self.assertFalse(admission.LOCK.exists())
        self.assertFalse(admission.OUT.exists())

    def test_analytic_engine_transport_and_three_renders_fail_incomplete_pack(self):
        # Temporary analytic sine tensor, never Mike inference or research evidence.
        tensor=(.1*np.sin(np.arange(8000)*2*np.pi*220/16000)).astype(np.float32)[None,None,None,:]
        class Session:
            def run(self,*args,**kwargs):return [tensor]
        fake=types.SimpleNamespace(SessionOptions=types.SimpleNamespace,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
            InferenceSession=lambda *args,**kwargs:Session())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model';model.write_bytes(b'analytic fixture only')
            reservation=root/'reservation.json';lock=root/'lock.json';out=root/'out'
            reservation.write_text(json.dumps(dict(source_sha256=admission.sha(admission.__file__),
                versions={},files={},model=dict(model_path=str(model),config_path=str(model),
                expected_sha256=admission.sha(model),config_sha256=admission.sha(model),native_rate=16000),
                requests=[dict(id='analytic-fixture',token_ids=[1,2],length_scale=1.)])))
            lock.write_text(json.dumps(dict(source_sha256=admission.sha(admission.__file__),
                files={str(reservation):admission.sha(reservation)})))
            with patch.multiple(admission,LOCK=lock,RESERVATION=reservation,OUT=out), \
                 patch.object(admission,'versions',return_value={}), \
                 patch.object(admission.subprocess,'check_output',return_value=lock.read_bytes()), \
                 patch.dict(sys.modules,{'onnxruntime':fake}),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:admission.generate()
            self.assertEqual(result.exception.code,1)
            native=json.loads((out/'native-records.jsonl').read_text())
            renders=[json.loads(line) for line in (out/'render-records.jsonl').read_text().splitlines()]
            self.assertTrue(native['audit']['passed'])
            self.assertEqual(len(renders),3)
            self.assertTrue(all(row['audit']['passed'] for row in renders))
            self.assertEqual(np.load(out/'analytic-fixture.engine.npy').tobytes(),tensor.tobytes())
            self.assertFalse(json.loads((out/'summary.json').read_text())['mechanically_qualified'])


if __name__=='__main__':unittest.main()
