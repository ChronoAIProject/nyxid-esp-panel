"""Analytic adversarial counts, never speech or independent evidence."""
import copy
import json
from pathlib import Path
from select_melo_joint_development_v2 import joint_select,STRATA,C161
from melo_synthetic_viability_v2 import assess

def table(name):
 return dict(role='unconsumed-threshold-development',candidate_id=name,model_sha256=C161 if name=='c161' else '1'*64,population_sha256='2'*64,background_manifest_sha256='3'*64,background_hours=7.902862222222223,expected_positive_strata=list(STRATA),points=[dict(threshold=k/1280,positive={s:dict(clips=n,hits=n) for s,n in STRATA.items()},confusion_clips=288,confusion_activated_clips=0,background_events=0) for k in range(1,1281)])
def main():
 a=table('c161');b=table('speecht5-final20000')
 assert joint_select({'c161':a,'speecht5-final20000':b})['selected_candidate']=='c161'
 ref=copy.deepcopy(a['points'][0]); candidate=copy.deepcopy(ref)
 # Nonzero confusion can pass; the next integer above2.5pp cannot.
 candidate['confusion_activated_clips']=7
 assert assess(candidate,ref,ref,7.902862222222223)['passed']
 candidate['confusion_activated_clips']=8
 assert not assess(candidate,ref,ref,7.902862222222223)['passed']
 candidate['confusion_activated_clips']=0
 acceptance_point=copy.deepcopy(candidate)
 acceptance_point['positive']={s.replace('EN-US','EN-INDIA').replace('EN-BR','EN-AU'):v for s,v in acceptance_point['positive'].items()}
 assert not assess(acceptance_point,acceptance_point,acceptance_point,5,acceptance=True)['passed']
 assert assess(acceptance_point,acceptance_point,acceptance_point,6,acceptance=True)['passed']
 for p in a['points']:
  for s,n in STRATA.items():p['positive'][s]['hits']=8 if n==9 else 24
 for p in b['points']:p['positive']['EN-US/bare']['hits']=7
 result=joint_select({'c161':a,'speecht5-final20000':b})
 assert result['selected_candidate']=='c161' and result['selected_threshold']==1
 b['population_sha256']='4'*64
 try:joint_select({'c161':a,'speecht5-final20000':b})
 except ValueError as e:assert 'population' in str(e)
 else:raise AssertionError('Population mismatch accepted')
 report=dict(schema='melo-joint-selector-analytic-check-v2',fixtures='Artificial count tables only; never speech/model/acceptance evidence',exact_tie_prefers_c161=True,worst_voice_priority_over_higher_overall=True,higher_threshold_tie_rule=True,cross_model_population_mismatch_rejected=True,nonzero_confusion_within_margin_passes=True,integer_above_confusion_margin_fails=True,insufficient5hour_acceptance_fails=True,separate_acceptance_voice_strata_pass_at6hours=True,real_development_scores_consumed=False,promotion_qualified=False)
 output=Path('research/wakeword_p4/results/4090-melo-joint-selector-analytic-check-v2.json')
 output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
