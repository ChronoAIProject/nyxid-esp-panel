"""Compare prospective background to known input metadata; never open finals audio."""
import hashlib
import json
from pathlib import Path
import re
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
INPUTS=[
 BASE/'multispeaker-coverage-features-v1/broad/ambient.json',
 BASE/'multispeaker-coverage-features-v1/broad/human.json',
 BASE/'causal-onset-material-v1/records.jsonl',
 BASE/'causal-existing-nearword-features-v1/records.json',
 BASE/'hey-common-carrier-features-v1/features.json',
 BASE/'hey-broad-training-features-v1/hey_nyx/features.json',
 BASE/'speecht5-training-features-v1/features.json',
 R/'4090-voxconverse-dev0-audit-v1.json',R/'4090-voxconverse-dev1-audit-v1.json',
 R/'4090-voxconverse-new-development-dev2-audit-v1.json',R/'4090-voxconverse-new-development-dev3-audit-v1.json',
 R/'4090-hiesp-independent-negative-pcm-manifest-v1.json',
 R/'4090-v30-independent-paired-pcm-manifest-v1.json',R/'4090-ami-independent-input-manifest-v1.json']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory(value):
 hashes=set();names=set()
 def walk(v,key=''):
  if isinstance(v,dict):
   for k,w in v.items():walk(w,k)
  elif isinstance(v,list):
   for w in v:walk(w,key)
  elif isinstance(v,str):
   if 'sha256' in key and re.fullmatch('[0-9a-f]{64}',v):hashes.add(v)
   if 'path' in key and v.endswith(('.wav','.flac')):names.add(Path(v).stem)
 walk(value);return hashes,names
def main():
 path=BASE/'prospective-background-whole-waveform-v1/manifest.json';manifest=json.loads(path.read_text());rows=manifest['records'];assert len(rows)==42 and manifest['integrity_preparation_passed']
 hashes={r[k] for r in rows for k in ['sha256','pcm_sha256']};names={Path(r['path']).stem for r in rows};assert len(hashes)==84 and len(names)==42
 comparisons=[]
 for source in INPUTS:
  if source.suffix=='.jsonl':
   hs=set();ns=set()
   with source.open() as f:
    for line in f:
     h,n=inventory(json.loads(line));hs.update(h);ns.update(n)
  else:hs,ns=inventory(json.loads(source.read_text()))
  comparisons.append(dict(path=str(source),sha256=sha(source),known_hashes=len(hs),known_audio_ids=len(ns),hash_overlap=sorted(hashes&hs),recording_id_overlap=sorted(names&ns)))
 all_clear=all(not r['hash_overlap'] and not r['recording_id_overlap'] for r in comparisons)
 result=dict(schema='prospective-background-known-role-audit-v1',script_sha256=sha(__file__),manifest_sha256=sha(path),recordings=42,source_samples=sum(r['samples'] for r in rows),source_hours=sum(r['samples'] for r in rows)/16000/3600,known_identity_checks_passed=all_clear,comparisons=comparisons,final_audio_accessed=False,acceptance_downloaded=False,wake_scored=False,scoring_authorized=False,
  coverage='Actual candidate sampler source metadata, new SpeechT5 training metadata, four consumed VoxConverse development shard audits and three historical final manifests. Full-source hashes/audio IDs where already recorded; hash equality tested conservatively across any recorded sha256 field. No feature arrays or protected/final waveforms opened.',
  limits='Known whole-record encoded/PCM/hash and recording-ID overlap only. Metadata without raw PCM hashes cannot exclude transformed/subrecord duplicates. No claim of universal all-machine, pretrained-person or upstream editing independence; reserved test0..3 and acceptance test6/7 audio remain unopened. Before scoring, bind these explicit coverage limits in complete candidate/source/threshold/acceptance protocol, rather than infer proof of globally absent overlap.')
 with (R/'4090-prospective-background-known-role-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}),flush=True)
 if not all_clear:raise RuntimeError('Known role overlap retained; do not qualify sources')
if __name__=='__main__':main()
