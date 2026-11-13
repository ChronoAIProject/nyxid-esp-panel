"""Analytic admission tests only; never synthesize or score acceptance audio."""
import copy
import sys
import unittest
from unittest.mock import patch

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


if __name__=='__main__':unittest.main()
