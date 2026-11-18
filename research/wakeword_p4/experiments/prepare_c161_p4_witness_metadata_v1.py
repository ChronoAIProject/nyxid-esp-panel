"""Metadata-only handoff of existing consumed synthetic witnesses; no synthesis."""
import hashlib,json,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 plan=R/'4090-willow-mac-matched-prime-plan-v1.json'
 comparison=R/'4090-willow-mac-matched-prime-c161-comparison-v1.json'
 handoff=R/'4090-c161-mac-development-handoff-v1.json'
 p,c,h=map(read,[plan,comparison,handoff]);rows=[]
 assert len(p['records'])==len(c['records'])==12
 assert c['willow_plan_sha256']==sha(plan)
 for source,saved in zip(p['records'],c['records'],strict=True):
  assert source['pcm_sha256']==saved['source_pcm_sha256']
  assert all(source[k]==saved[k] for k in ['voice','target','form'])
  assert sha(source['path'])==source['sha256']
  rows.append(dict(index=saved['index'],voice=source['voice'],target=source['target'],form=source['form'],text=source['text'],wav_sha256=source['sha256'],source_pcm_sha256=source['pcm_sha256'],samples=source['samples'],original_mac_source=source['source_path'],original_mac_source_sha256=source['source_sha256'],historical_c161_events=saved['c161_events'],historical_official_willow_events=saved['willow_events']))
 model=Path(h['source_path']);assert sha(model)==h['model']['sha256'] and model.stat().st_size==60952
 out=dict(schema='c161-p4-consumed-witness-metadata-v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources={str(f):sha(f) for f in [plan,comparison,handoff,Path(__file__)]},model=h['model'],model_gzip=h['gzip'],profile=h['profile'],witnesses=rows,willow_reference=dict(model='wn9_heywillow_tts',threshold_bits='3f24dd2f',threshold_mode='DEFAULT',total_prime_samples=81600,native_call_samples=512),p4_results=None,limits='Metadata only; no PCM supplied or regenerated, no new inference or exposure. All12 sources already consumed development material. Historical event clocks differ between detectors. Missing original source means source unavailable, not permission to synthesize and retain its old hash. P4 numerical injection, acoustic playback, command retention, whole-device resources and stability are separate unproven requirements. No threshold or candidate selection.')
 with (R/'4090-c161-p4-consumed-witness-metadata-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print('Verified and preserved12 original source identities; no inference or training')
if __name__=='__main__':main()
