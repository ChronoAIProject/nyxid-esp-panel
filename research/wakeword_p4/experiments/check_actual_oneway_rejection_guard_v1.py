"""Actual rejected screen cannot pass admission even with optimistic reference fixture."""
import copy,json,hashlib
from pathlib import Path
from admit_hey_broad_oneway_consistency_numerical_v1 import assess
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-result-v1.json';cases=json.loads(p.read_text())['cases'];op=R/'4090-hey-broad-trained-quantization-plan-v1.json';orders=[r['order'] for r in json.loads(op.read_text())['requests']]
 # Explicitly fabricated optimistic reference: no actual recursive result exists.
 fixture=[dict(target=c['target'],condition=c['condition'],request_order=c['request_order'],actual_output_exact=True,twice_output_state_exact=True,all_encoder_fc_exact=True,metrics_both_repeats={'fixture':dict(unequal=0,maximum=0)}) for c in cases]
 checks=[]
 for condition in ['twoway_consistency','oneway_consistency']:
  for falseflags in [False,True]:
   rows=copy.deepcopy(cases)
   if falseflags:
    for c in rows:c['passed']=True
   try:assess(rows,fixture,orders,condition)
   except ValueError as e:
    assert 'Original wholewave numerical gate rejected' in str(e)
    checks.append(dict(condition=condition,forged_pass_flags=falseflags,rejected=True,reason=str(e)))
   else:raise AssertionError('Actual failed screen admitted')
 assert not (R/'4090-hey-broad-oneway-consistency-numerical-admission-v1.json').exists()
 out=dict(screen_sha256=sha(p),admission_source_sha256=sha(Path(__file__).with_name('admit_hey_broad_oneway_consistency_numerical_v1.py')),checks=checks,reference_is_constructed_optimistic_fixture=True,actual_recursive_evidence_available=False,fresh_material_reads=0,model_invocations=0,admission_created=False,limits='Negative guard test only. Constructed reference is not real recursive evidence; rejection occurs on actual screen numbers before reference acceptance. No current candidate qualification.')
 with (R/'4090-actual-oneway-rejection-guard-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
