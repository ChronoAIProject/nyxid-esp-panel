"""Constructed raw-byte cases only, no development score inspection."""
import unittest
import numpy as np
from hey_development_metrics_v1 import GRID,event_samples,far_upper,synthetic_metrics,eligible,choose


def corpus():
    clips=[]
    for target in ('hey_nyx','hey_willow','hi_esp')+('common_negative',)*7:
        for carrier in ('','a','b','c'):
            for speed in (.9,1.,1.1):
                for variant in range(3):
                    row=dict(target_id=target,samples=50000,lead_samples=16000,generated_prefix_end_sample=32000,full_rir_prefix_end_sample=33232,generated_container_end_sample=34000,full_rir_container_end_sample=35232,carrier=carrier,length_scale=speed,variant=variant,voice='fixture',source_id='fixture')
                    clips.append((row,np.zeros(100,dtype=np.uint8)))
    return clips


class Metrics(unittest.TestCase):
    def test_complete_phrase_not_early_activation(self):
        clips=corpus()
        for row,raw in clips:
            if row['target_id']=='hey_nyx':raw[40:45]=255
        m=synthetic_metrics(clips,'hey_nyx',.5)
        self.assertEqual(m['any_after_source_recall'],1)
        self.assertEqual(m['macro'],0)
        self.assertFalse(eligible(m,.3))
        for row,raw in clips:
            raw[:]=0
            if row['target_id']=='hey_nyx':raw[75:80]=255
        m=synthetic_metrics(clips,'hey_nyx',.5)
        self.assertEqual(m['macro'],1);self.assertTrue(eligible(m,.3))
        clips[-1][1][75:80]=255
        m=synthetic_metrics(clips,'hey_nyx',.5)
        self.assertFalse(eligible(m,.3));self.assertEqual(m['negatives']['common']['events'],1)

    def test_clock_far_and_rank(self):
        self.assertEqual(len(GRID),138)
        self.assertEqual(event_samples(np.full(5,255,np.uint8),.5,1000),[(5,1000)])
        self.assertAlmostEqual(far_upper(0,10),.299573227355399,places=12)
        self.assertGreater(far_upper(1,10),far_upper(0,10))
        self.assertIsNone(choose([]))
        base=dict(eligible=True,synthetic={'macro':.9},background_upper95=.3)
        self.assertEqual(choose([dict(base,threshold=.8),dict(base,threshold=.9)])['threshold'],.9)
        for args in ((-1,10),(0,0)):
            with self.assertRaises(ValueError):far_upper(*args)

if __name__=='__main__':unittest.main()
