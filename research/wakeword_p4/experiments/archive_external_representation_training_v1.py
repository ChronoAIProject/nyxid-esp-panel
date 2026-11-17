"""Preserve completed original arm bytes; independently verify every archive member."""
import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/external-representation-training-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(arm):
 assert arm in ['control','feature']
 report=R/f'4090-external-representation-{arm}-training-original-v1.json';d=json.loads(report.read_text());assert d['steps']==d['optimizer_updates']==20000
 assert d['launch']['arm']==arm and d['cached_targets_unchanged']
 for name,h in d['files'].items():assert sha(ROOT/arm/name)==h,name
 files={f'arm/{name}':ROOT/arm/name for name in d['files']}
 files['training-original.json']=report
 files['training-plan.json']=R/'4090-external-representation-training-plan-v1.json'
 log=ROOT/f'{arm}-execution.log';files['execution.log']=log
 manifest=dict(schema='external-representation-original-training-archive-v1',arm=arm,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),members={name:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for name,p in files.items()},limits='Original completed-arm bytes only. Checkpoint includes training-only projection and optimizer; not a deployment model or independent training/recognition qualification. Actual exits recorded separately.')
 raw=(json.dumps(manifest,indent=2)+'\n').encode();dest=R/f'4090-external-representation-{arm}-training-evidence-v1.tar.gz'
 with tarfile.open(dest,'x:gz') as t:
  for name,p in files.items():t.add(p,arcname=name)
  member=tarfile.TarInfo('manifest.json');member.size=len(raw);t.addfile(member,io.BytesIO(raw))
 with tarfile.open(dest,'r:gz') as t:
  names=t.getnames();assert len(names)==len(set(names)) and set(names)==set(files)|{'manifest.json'}
  assert t.extractfile('manifest.json').read()==raw
  for name,spec in manifest['members'].items():
   b=t.extractfile(name).read();assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256']
 out=dict(schema='external-representation-original-training-archive-review-v1',arm=arm,all_members_verified=True,verified_members=len(files)+1,archive=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)),manifest=manifest)
 with (R/f'4090-external-representation-{arm}-training-archive-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(arm=arm,verified_members=len(files)+1,archive=out['archive'])))
if __name__=='__main__':main(sys.argv[1])
