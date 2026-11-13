"""Reject incomplete or mismatched formal input evidence before new development."""
import json
import unittest

import score_recovered_new_development_v1 as executor


class MatchedAdmission(unittest.TestCase):
    def test_full_original_schedule_required(self):
        expected=json.loads((executor.ROOT/'results/4090-per-draw-noise-full-support-v1.json').read_text())
        audit=dict(steps_per_arm=120000,different_main_network_batches=120000,
                   original_ordered_paired_source_draws_exact=True,paired_network_inputs_exact=True,
                   initialization_inventory_exact=True,schedule_sha256=expected['schedule_sha256'],
                   acceptance_claim=False)
        executor.validate_matched_input_audit(audit)
        for key,value in [('steps_per_arm',119999),('different_main_network_batches',0),
                          ('original_ordered_paired_source_draws_exact',False),
                          ('paired_network_inputs_exact',False),('initialization_inventory_exact',False),
                          ('schedule_sha256','0'*64),('acceptance_claim',True)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                executor.validate_matched_input_audit(dict(audit,**{key:value}))

    def test_actual_unfinished_run_is_not_ready(self):
        required,missing=executor.prerequisites()
        self.assertIn(str(executor.BASE/'complete-recovery-formal-input-audit-v1.json'),missing)
        self.assertFalse(executor.FREEZE.exists())
        self.assertFalse(executor.OUT.exists())
        with self.assertRaisesRegex(RuntimeError,'incomplete'):executor.write_freeze()
        self.assertFalse(executor.FREEZE.exists())
        self.assertFalse(executor.OUT.exists())


if __name__=='__main__':unittest.main()
