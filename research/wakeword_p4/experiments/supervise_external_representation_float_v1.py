"""Sequential fixed full float attribution, preserve actual exits and raw traces."""
import json,os,subprocess,time,tarfile,traceback,hashlib
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/external-representation-float-supervision-v1');LOCK=R/'4090-external-representation-float-supervision-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 return p
def main():
 verify();B.mkdir(exist_ok=False);state=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],status='running',plan_sha256=sha(LOCK),stages=[],archives={})
 def save():(B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 def stage(name,args):
  verify();path=B/(name+'.log');start=time.monotonic()
  with path.open('x') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  row=dict(name=name,args=args,actual_exit_code=p.returncode,elapsed_seconds=time.monotonic()-start,log=str(path),log_sha256=sha(path));state['stages'].append(row);save();print(json.dumps(row),flush=True);assert p.returncode==0,row
 def publish(paths,message):
  subprocess.run(['git','add','--',*[str(p) for p in paths]],check=True);subprocess.run(['git','commit','-m',message],check=True);subprocess.run(['git','push','origin','research-hye-nyx-voice-gpu'],check=True)
 try:
  save()
  for arm in ['control','feature']:
   stage('float-'+arm,['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py','research/wakeword_p4/experiments/replay_external_representation_float_v1.py','run',arm])
   root=Path(f'.local/wakeword/external-representation-{arm}-float-replay-v1');files=sorted(p for p in root.rglob('*') if p.is_file());members={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};dest=R/f'4090-external-representation-{arm}-float-evidence-v1.tar.gz'
   with tarfile.open(dest,'x:gz') as t:
    for p in files:t.add(p,arcname=str(p.relative_to(root)))
   with tarfile.open(dest,'r:gz') as t:
    names=t.getnames();assert len(names)==len(set(names)) and set(names)==set(members)
    for m in t:
     data=t.extractfile(m).read();assert len(data)==members[m.name]['bytes'] and hashlib.sha256(data).hexdigest()==members[m.name]['sha256']
   state['archives'][arm]=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),members=members);save()
   manifest=R/f'4090-external-representation-{arm}-float-archive-review-v1.json';manifest.write_text(json.dumps(state['archives'][arm],indent=2)+'\n')
   publish([dest,manifest,R/f'4090-external-representation-{arm}-float-replay-original-v1.json'],'research: preserve full '+arm+' float attribution and verified archive')
  stage('independent-float-review',['/usr/bin/python3','research/wakeword_p4/experiments/review_external_representation_float_v1.py'])
  publish([R/'4090-external-representation-float-review-v1.json'],'research: preserve independent full float event review')
  state['status']='completed-await-research-decision'
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc()
 finally:
  save();dest=R/'4090-external-representation-float-supervision-original-v1.json';dest.write_text(json.dumps(state,indent=2)+'\n');publish([dest],'research: preserve external feature float execution receipts')
 if state['status']!='completed-await-research-decision':raise SystemExit(1)
if __name__=='__main__':main()
