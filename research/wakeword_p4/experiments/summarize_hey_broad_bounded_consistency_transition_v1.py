"""Descriptive paired outcome transition; no model/data access or selection."""
import json,hashlib
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-hey-broad-bounded-consistency-fixed-grid-screen-result-v1.json';a=R/'4090-hey-broad-bounded-consistency-fixed-grid-screen-independent-audit-v1.json'
 data=json.loads(p.read_text());audit=json.loads(a.read_text());assert audit['result_sha256']==sha(p)
 rows=data['cases'];assert len(rows)==540;summary={}
 for target in ('hey_willow','hey_nyx'):
  by={condition:{c['request_order']:c for c in rows if c['target']==target and c['condition']==condition} for condition in ('raw_logit_control','bounded_logit_consistency')}
  control=by['raw_logit_control'];treatment=by['bounded_logit_consistency'];assert len(control)==len(treatment)==135 and control.keys()==treatment.keys()
  states={}
  for name,left,right in [('both_pass',True,True),('resolved',False,True),('introduced',True,False),('both_fail',False,False)]:states[name]=sorted(i for i in control if control[i]['passed']==left and treatment[i]['passed']==right)
  assert sum(map(len,states.values()))==135
  summary[target]=dict(requests=states,counts={k:len(v) for k,v in states.items()},control_passed=sum(c['passed'] for c in control.values()),treatment_passed=sum(c['passed'] for c in treatment.values()))
 out=R/'4090-hey-broad-bounded-consistency-paired-transition-v1.json'
 with out.open('x') as f:json.dump(dict(files={str(f):sha(f) for f in (Path(__file__).resolve().relative_to(Path.cwd()),p,a)},summary=summary,qualified_complete270_conditions=audit['qualified_complete270_conditions'],model_invocations=0,fresh_data_reads=0,limits='Complete original paired numerical outcomes only. Same original initialization/sampler and fixed conditions; no threshold/checkpoint/condition selection. Net pass-count improvement is not admission, recognition/FAR evidence or unique causal attribution. Both conditions rejected; original failures retained.'),f,indent=2);f.write('\n')
 print(json.dumps({t:r['counts'] for t,r in summary.items()}))
if __name__=='__main__':main()
