"""Synthetic event ledger tests only; no development inference."""
import unittest
from calibrate_development import metrics


def corpus(target,at):
    rows=[]
    for kind in ('hey_nyx','hey_willow','hi_esp')+('common_negative',)*7:
        for carrier in ('','a','b','c'):
            for speed in (.9,1.,1.1):
                for variant in range(3):
                    row=dict(target_id=kind,source_id='fixture',variant=variant,carrier=carrier,length_scale=speed,lead_samples=16000,generated_prefix_end_sample=32000,full_rir_prefix_end_sample=33232,generated_container_end_sample=34000,full_rir_container_end_sample=35232)
                    rows.append((row,[dict(call=70,sample_end=at,result=1)] if kind==target else []))
    return rows


class Metrics(unittest.TestCase):
    def test_same_word_and_hiesp_labels(self):
        for model,target,cross in [('willow','hey_willow',36),('hiesp','hi_esp',72)]:
            m=metrics(corpus(target,35000),model,True)
            self.assertEqual(m['macro'],1);self.assertEqual(m['negatives']['cross_target']['clips'],cross)
            positive=next(r for r in m['records'] if r['target_id']==target)
            self.assertEqual(positive['events'][0]['generated_prefix_end_sample_signed_ms'],187.5)
            self.assertEqual(m['full_rir_prefix_recall'],1)
    def test_early_event_cannot_be_full_phrase_hit(self):
        m=metrics(corpus('hi_esp',20000),'hiesp')
        self.assertEqual(m['macro'],0);self.assertEqual(m['any_after_source_recall'],1)
        m=metrics(corpus('hey_willow',10000),'willow')
        self.assertEqual(m['pre_source_positive_events'],36);self.assertEqual(m['macro'],0)

if __name__=='__main__':unittest.main()
