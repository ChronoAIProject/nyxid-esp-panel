"""Exhaustive endpoint mapping and existing feature binary provenance check."""
import sys,json,hashlib,collections
from pathlib import Path
sys.path.insert(0,'research/wakeword_p4/training')
from generation_reference_support import GenerationReferenceSupport
base=Path('research/wakeword_p4/results');p=base/'4090-training-reference-endpoint-feasibility-v1.json';expected='11a3a32230c52602eb100a54304ebffe29c2551a30fa3733af34cee6f8f59b79';support=GenerationReferenceSupport(p,expected)
root=Path('.local/wakeword/features-gpu-ami-extra-v1');rows=json.loads((root/'features.json').read_text());counts=collections.Counter()
for i,s in support.rows.items():
 r=rows[i];interval=s['target_endpoints']
 for j in range(3,28):
  original=min(r['frames'],max(150,r['keyword_frame']+j))
  c,_=support.choose(i,r,original,False);t,reason=support.choose(i,r,original,True);assert c==original
  if interval['count']:
   assert interval['first']<=t<=interval['last']
   assert abs(t-original)==min(abs(interval['first']-original),abs(interval['last']-original)) if not interval['first']<=original<=interval['last'] else t==original
  else:assert t==original
  counts[reason]+=1
pbin=root/'features.bin';h=hashlib.sha256()
with pbin.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
expected_binary=json.loads((root/'base-provenance.json').read_text())['base_feature_binary_sha256'];assert h.hexdigest()==expected_binary
assert pbin.stat().st_size==sum(r['frames'] for r in rows)*80
out={'scope':'All original25 main-positive candidates per63300 rows; deterministic mapping only, no RNG or model calls.','ledger_sha256':expected,'module_sha256':hashlib.sha256(Path('research/wakeword_p4/training/generation_reference_support.py').read_bytes()).hexdigest(),'candidate_counts':dict(counts),'feature_binary_bytes':pbin.stat().st_size,'feature_binary_sha256':h.hexdigest(),'matches_existing_base_provenance':True,'control_unchanged':True,'unsupported_fallback_unchanged':True,'nearest_legal_endpoint_verified':True,'limits':'Whole existing binary hash verification only, no dev/final inference or content selection. Does not independently reconstruct rendering/frontend or establish model quality. No optimizer updates.'}
(base/'4090-generation-reference-mapping-preflight-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
