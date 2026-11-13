"""Actual original sampler versus v2 on analytic features; no TF/audio/model."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_hey_training_batch_v1 import audit
from audit_hey_training_batch_v2 import audit as audit_v2
from hey_prefix_crop_v2 import CropRule,apply_crop_rule,effective_jitter


class CropChecks(unittest.TestCase):
    def test_exhaustive_draws_and_invalid_support(self):
        for span in range(1,148):
            row=dict(keyword_frame=178+span,lead_frame=178)
            for draw in range(3,28):
                actual=effective_jitter(row,draw)
                self.assertEqual(actual,min(draw,27,150-span))
                self.assertLessEqual(row['keyword_frame']+actual-150,row['lead_frame'])
        with self.assertRaises(ValueError):effective_jitter(dict(keyword_frame=326,lead_frame=178),3)
        with self.assertRaises(ValueError):effective_jitter(dict(keyword_frame=304,lead_frame=178,positive_jitter_max=27),3)

    def test_actual_batches_keep_rng_labels_weights_negatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            raw=(np.arange(1000*40).reshape(1000,40)%600).astype('<u2');raw.tofile(root/'features.bin')
            rows=[dict(split='training',label=label,engine='analytic',offset=label*500,frames=500,
                       lead_frame=178,keyword_frame=304,positive_jitter_max=24,source_id='fixture',variant=2)
                  for label in (0,1)]
            (root/'features.json').write_text(json.dumps(rows))
            np.save(root/'ambient.npy',np.zeros((500,40),dtype=np.uint16))
            (root/'human.json').write_text('[]')
            (root/'ambient.json').write_text(json.dumps([dict(split='training',path='ambient.npy')]))
            original=audit(root,return_sampler=True)
            changed=audit_v2(root,return_sampler=True,trace_path=root/'crop-trace.jsonl')
            rule=changed['crop_rule']
            different=0
            for _ in range(10):
                x,y,w=original['batch']();xx,yy,ww=changed['batch']()
                np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(w,ww)
                np.testing.assert_array_equal(x[y[:,0]==0],xx[y[:,0]==0])
                different+=int(not np.array_equal(x,xx))
                for key in ('rng','mining_rng','augmentation_rng'):
                    self.assertEqual(original[key].bit_generator.state,changed[key].bit_generator.state)
            result=rule.finish();self.assertEqual(result['positive_draws'],320)
            self.assertGreater(result['capped_draws'],0);self.assertGreater(different,0)
            saved=list(map(json.loads,(root/'crop-trace.jsonl').read_text().splitlines()))
            self.assertEqual(len(saved),320)
            self.assertTrue(all(r['effective_jitter']==min(r['original_jitter'],24) for r in saved))


if __name__=='__main__':unittest.main()
