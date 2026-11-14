"""Numerical intake contract for a locally authenticated original Mac capture receipt."""
import hashlib,json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(receipt,expected,software_root):
 assert receipt['actual_trained_microinterpreter_run'] is True
 # Source report remains independently authenticated by the importer; not proven by flags.
 original=Path(receipt['original_report_path']);assert sha(original)==receipt['original_report_sha256']
 assert receipt['inputs_sha256']=='0f7eff41940be4149be682f9a24755d42646bbe08e211c5fbdd1638059d3c9ac'
 assert len(receipt['models'])==len(expected['cases'])==4
 files={str(original):sha(original)};seen=set()
 for entry in receipt['models']:
  key=entry['target'],entry['condition'];assert key not in seen;seen.add(key)
  r=next(x for x in expected['cases'] if (x['target'],x['condition'])==key);assert entry['model_sha256']==r['model_sha256']
  path=Path(entry['probability_npz']);assert sha(path)==entry['probability_npz_sha256'];files[str(path)]=sha(path)
  reference=Path(software_root)/(key[0]+'-'+key[1])/'references.npz';assert sha(reference)==r['references_sha256'];files[str(reference)]=sha(reference)
  with np.load(path,allow_pickle=False) as captured,np.load(reference,allow_pickle=False) as refs:
   assert captured.files==['probability'];v=captured['probability'];assert v.shape==(4,2,256) and v.dtype==np.float32 and np.isfinite(v).all()
   assert v[:,0].tobytes()==v[:,1].tobytes()
   for backend in ['tf','litert']:assert np.max(np.abs(v.astype(float)-refs[backend].astype(float)))<=1e-4
 assert seen=={(x['target'],x['condition']) for x in expected['cases']}
 return files
def reject_flag_only():
 try:verify(dict(actual_trained_microinterpreter_run=True,all_passed=True),dict(cases=[]),'.')
 except (KeyError,AssertionError):return True
 raise AssertionError('Flag-only receipt accepted')
if __name__=='__main__':
 assert reject_flag_only()
 print(json.dumps(dict(flag_only_rejected=True,actual_mac_capture_checked=False,limits='Receipt contract preflight only; actual capture intake awaits original Mac output.')))
