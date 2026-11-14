"""Observe original official sequence only; publish existing complete calibration."""
import json,hashlib,time,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results');P=R/'4090-original-official-calibration-publication-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(pid):
 p=Path(f'/proc/{pid}/stat')
 if not p.exists():return None
 f=p.read_text().rsplit(') ',1)[1].split()
 return dict(state=f[0],starttime=f[19],cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().hex())
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();plan=json.loads(P.read_text())
 out=Path('.local/wakeword/original-official-calibration-publication-v1');out.mkdir(exist_ok=False)
 try:
  for p,h in plan['files'].items():assert sha(p)==h,p
  while True:
   current=ident(plan['pid'])
   if current is None or current['state']=='Z':break
   assert current['starttime']==plan['identity']['starttime'] and current['cmdline']==plan['identity']['cmdline']
   time.sleep(5)
  for p,h in plan['files'].items():assert sha(p)==h,p
  execution=json.loads(Path(plan['execution']).read_text());root=Path(execution['output']);done=json.loads((root/'complete.json').read_text());assert done['streams']==9672 and done['execution_sha256']==sha(plan['execution']) and done['retries']==0
  source=root/'calibration.json';d=json.loads(source.read_text());assert d['execution_sha256']==sha(plan['execution']) and set(d['curves'])=={'willow','hiesp'} and all(len(v)==13 for v in d['curves'].values())
  assert d['threshold_selected']==d['acceptance_authorized']==d['candidate_promoted']==False
  dest=R/'4090-original-official-development-calibration-v1.json'
  with dest.open('xb') as f:f.write(source.read_bytes())
  receipt=R/'4090-original-official-calibration-publication-result-v1.json'
  with receipt.open('x') as f:json.dump(dict(source=str(source),source_sha256=sha(source),plan_sha256=sha(P),complete_sha256=sha(root/'complete.json'),status='published_original_bytes_no_new_inference',limits='Producer audits all9672 streams. Publisher checks completeness/identity only, not independent re-audit or acceptance.'),f,indent=2);f.write('\n')
  assert not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()
  subprocess.run(['git','add','--',str(dest),str(receipt)],check=True);subprocess.run(['git','commit','-m','research: publish original complete official development calibration'],check=True);subprocess.run(['git','push','origin','HEAD:research-hye-nyx-voice-gpu'],check=True)
 except Exception as e:
  (out/'failure.json').write_text(json.dumps(dict(status='stopped_no_retry',error=repr(e),plan_sha256=sha(P)),indent=2)+'\n');raise
if __name__=='__main__':run()
