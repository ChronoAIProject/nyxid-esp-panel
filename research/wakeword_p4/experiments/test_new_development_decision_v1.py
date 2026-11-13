"""Analytic event-trace counterexamples for preregistered decision semantics."""
import copy
import unittest

import numpy as np
from scipy.stats import chi2

from calibrate_recovered_new_development_v1 import synthetic_metrics,eligible,choose,GRID


def fixtures(positive_index=55):
    """Artificial score arrays only; no model inference or synthesized audio."""
    rows=[]
    for target,texts in [('hey_nyx',['Hey Nicks.','Hey Nicks!','Hey Nicks, time?','Hey Nicks, light.']),
                         ('hi_esp',['Hi E S P.','Hi E S P!','Hi E S P, time?','Hi E S P, light.']),
                         ('confusion',[f'Other {i}.' for i in range(12)])]:
        for text in texts:
            for speed in [.9,1.,1.1]:
                for draw in range(2):
                    for variant in range(3):
                        scores=np.zeros(110)
                        if target=='hey_nyx':scores[positive_index]=1.
                        row=dict(target=target,text=text,length_scale=speed,variant=variant,lead_samples=28800,
                                 path=f'{target}-{text}-{speed}-{draw}-{variant}',
                                 generated_output_end_sample=40000,full_rir_end_sample=40000+(960 if variant==2 else 0))
                        rows.append((row,scores))
    return rows


class DecisionSemantics(unittest.TestCase):
    def test_all_population_counts_and_perfect_activation(self):
        m=synthetic_metrics(fixtures(),.9,True)
        self.assertEqual((m['positive_clips'],m['activated'],len(m['records'])),(72,72,360))
        self.assertEqual((m['macro'],m['standalone'],m['command']),(1.,1.,1.))
        self.assertEqual(m['negatives']['common']['clips'],216)
        self.assertEqual(m['negatives']['cross_target']['clips'],72)

    def test_before_start_is_failure_even_with_later_positive_event(self):
        rows=fixtures(75)
        for row,scores in rows:
            if row['target']=='hey_nyx':scores[0]=1.
        m=synthetic_metrics(rows,.9)
        self.assertEqual(m['activated'],72)
        self.assertEqual(m['pre_source_positive_events'],72)
        self.assertFalse(eligible(m,.4,.8))

    def test_final_tail_activation_counts_and_signed_early_reference_is_kept(self):
        late=synthetic_metrics(fixtures(105),.9,True)
        self.assertEqual(late['activated'],72) # no invented end+.8 cutoff
        early=synthetic_metrics(fixtures(55),.9,True)
        self.assertLess(early['records'][0]['signed_generated_end_ms'][0],0)

    def test_one_negative_event_or_short_exposure_cannot_be_rescued(self):
        rows=fixtures();next(scores for row,scores in rows if row['target']=='confusion')[55]=1.
        m=synthetic_metrics(rows,.9)
        self.assertFalse(eligible(m,.4,.8))
        perfect=synthetic_metrics(fixtures(),.9)
        upper_short=float(.5*chi2.ppf(.95,2)/1.2048112673611107)
        self.assertGreater(upper_short,.5)
        self.assertFalse(eligible(perfect,upper_short,.8))

    def test_command_only_failure_and_null_no_fallback(self):
        rows=fixtures()
        removed=0
        for row,scores in rows:
            if row['target']=='hey_nyx' and ', ' in row['text'] and removed<10:
                scores[:]=0.;removed+=1
        m=synthetic_metrics(rows,.9)
        self.assertGreater(m['macro'],.8)
        self.assertEqual(m['command'],26/36)
        self.assertFalse(eligible(m,.4,.8))
        self.assertIsNone(choose([dict(threshold=.9,synthetic=m,background_upper95=.4,eligible=False)]))

    def test_exact_selection_priority_macro_then_far_then_higher_threshold(self):
        template=dict(synthetic=dict(macro=.9),background_upper95=.4,eligible=True)
        points=[dict(copy.deepcopy(template),threshold=.8),dict(copy.deepcopy(template),threshold=.9),
                dict(copy.deepcopy(template),threshold=.95)]
        points[2]['background_upper95']=.45
        self.assertEqual(choose(points)['threshold'],.9)
        points[0]['synthetic']['macro']=.95
        self.assertEqual(choose(points)['threshold'],.8)
        self.assertEqual((len(GRID),GRID[0],GRID[-1]),(138,.5,.99609375))


if __name__=='__main__':unittest.main()
