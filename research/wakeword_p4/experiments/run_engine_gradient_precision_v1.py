"""Run four isolated zero-update precision workers; never changes live training."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-engine-gradient-precision-execution-v1.json';RESULT=R/'4090-engine-gradient-precision-result-v1.json';OUT=Path('.local/wakeword/engine-gradient-precision-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def observe():
 path=Path('.local/wakeword/engine-adversarial-formal-v1/startup.json');d=json.loads(path.read_text());proc=Path('/proc')/str(d['pid']);present=proc.exists();start=(proc/'stat').read_text().split()[21] if present else None
 return dict(startup=d,process_present=present,actual_starttime=start,original_process_live=present and start==d['starttime'])
def main():
 if sys.argv[1]=='lock':
  old=R/'4090-engine-adversarial-gradient-execution-v1.json';d=json.loads(old.read_text());files={p:h for p,h in d['files'].items() if p.endswith('.py')};probe=R/'4090-engine-adversarial-gradient-result-v1.json';audit=R/'4090-engine-adversarial-gradient-independent-audit-v1.json';a=json.loads(audit.read_text());assert a['all_passed'] and a['original_result_sha256']==sha(probe);aux=R/'4090-engine-adversarial-input-result-v1.json'
  paths=[old,probe,audit,aux,Path(__file__),Path(__file__).with_name('probe_engine_gradient_precision_worker_v1.py'),Path(__file__).with_name('audit_engine_gradient_precision_v1.py'),Path(__file__).with_name('engine_gradient_precision_control_v1.md')]
  for target in json.loads(probe.read_text())['targets']:
   paths.extend([Path(target['cases'][0]['arrays_path']),Path(target['initialization_path'])]);init=json.loads(Path(target['initialization_path']).read_text());paths.extend([Path(init['initial_h5_path']),Path(init['state_path'])])
  paths.append(Path(json.loads(aux.read_text())['batches'][0]['arrays_path']))
  for p in paths:files[str(p)]=sha(p)
  for p,h in files.items():assert sha(p)==h,p
  with LOCK.open('x') as f:json.dump(dict(files=files,original_probe=str(probe),original_auxiliary=str(aux),modes=['gpu-default','cpu-default','gpu-off','cpu-off'],versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','h5py']},pointwise_rule=dict(atol=1e-4,rtol=1e-4),optimizer_updates=0),f,indent=2)
  return
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();d=json.loads(LOCK.read_text())
 for p,h in d['files'].items():assert sha(p)==h,p
 for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False);before=observe();reports=[]
 try:
  for mode in d['modes']:
   with (OUT/(mode+'.log')).open('x') as f:subprocess.run([sys.executable,'research/wakeword_p4/training/gpu_python.py','research/wakeword_p4/experiments/probe_engine_gradient_precision_worker_v1.py',mode],stdout=f,stderr=subprocess.STDOUT,check=True)
   p=OUT/mode/'report.json';report=json.loads(p.read_text());assert report['all_mechanics_passed'];reports.append(dict(mode=mode,path=str(p),sha256=sha(p)));print(json.dumps(dict(mode=mode,completed=True)),flush=True)
  after=observe()
  with RESULT.open('x') as f:json.dump(dict(execution_sha256=sha(LOCK),reports=reports,formal_before=before,formal_after=after,all_workers_completed=True,optimizer_updates=0,limits='Worker mechanics completion only; cross-mode numerical comparisons and original-default reproduction still require independent saved-array reduction. Live training process untouched.'),f,indent=2)
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
