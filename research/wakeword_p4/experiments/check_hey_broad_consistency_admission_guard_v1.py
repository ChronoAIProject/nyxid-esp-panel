"""Historical rejected models must remain rejected; no fresh data or model invoke."""
import copy,json,hashlib
from pathlib import Path
from admit_hey_broad_consistency_numerical_v1 import assess
R=Path('research/wakeword_p4/results')
def main():
 screen=R/'4090-hey-broad-fixed-grid-screen-result-v1.json';reference=R/'4090-hey-broad-fixed-grid-recursive-result-v1.json';original=R/'4090-hey-broad-trained-quantization-plan-v1.json';cases=json.loads(screen.read_text())['cases'];refs=json.loads(reference.read_text())['cases'];orders=[r['order'] for r in json.loads(original.read_text())['requests']];checks=[]
 def rejected(name,rows,actual,condition):
  try:assess(rows,actual,orders,condition)
  except ValueError as error:checks.append(dict(case=name,rejected=True,reason=str(error)));return
  raise AssertionError('Historical failed data admitted: '+name)
 for condition in ['float_control','integer_qat']:
  rejected(condition+'-original',cases,refs,condition)
  changed=copy.deepcopy(cases)
  for c in changed:c['passed']=True
  rejected(condition+'-false-pass-flags',changed,refs,condition)
 changed=copy.deepcopy(cases);selected=[i for i,c in enumerate(changed) if c['condition']=='integer_qat' and c['target']=='hey_willow'];changed[selected[-1]]=copy.deepcopy(changed[selected[0]]);rejected('duplicate-request-keeps-count',changed,refs,'integer_qat')
 out=R/'4090-hey-broad-consistency-admission-guard-check-v1.json'
 with out.open('x') as f:json.dump(dict(checks=checks,source_sha256=hashlib.sha256(Path(__file__).with_name('admit_hey_broad_consistency_numerical_v1.py').read_bytes()).hexdigest(),original_screen_sha256=hashlib.sha256(screen.read_bytes()).hexdigest(),fresh_material_reads=0,model_invocations=0,admission_receipt_created=False,limits='Actual historical failures + misleading pass flags/duplicate identities remain rejected. No current candidate qualification or fresh-development access.'),f,indent=2);f.write('\n')
 print(len(checks),'historical rejection checks passed')
if __name__=='__main__':main()
