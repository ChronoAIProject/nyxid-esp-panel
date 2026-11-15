"""Resume verified infrastructure checkpoint, never select by recognition."""
import argparse,gzip,hashlib,json,os,shutil,subprocess,sys,zlib
from pathlib import Path
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword');OLD=BASE/'matchbox-long-control-recovery-v1';NEW=BASE/'matchbox-long-control-resumed-v1';SCRIPT=Path('research/wakeword_p4/experiments/train_matchbox_long_v1.py');LOCK=R/'4090-matchbox-long-control-resume-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 import torch
 checkpoint=OLD/'recovery.pt';d=torch.load(checkpoint,map_location='cpu',weights_only=False);assert d['step']==3500 and d['arm']=='control'
 assert not (OLD/'final.pt').exists()
 launch=json.loads((OLD/'launch.json').read_text());assert not Path(f"/proc/{launch['pid']}").exists()
 dec=zlib.decompressobj(16+zlib.MAX_WBITS);raw=dec.decompress((OLD/'batches.jsonl.gz').read_bytes());records=[]
 for line in raw.splitlines():
  try:records.append(json.loads(line))
  except json.JSONDecodeError:pass
 assert len(records)>=3500 and [x['step'] for x in records[:3500]]==list(range(1,3501))
 paths=[Path(__file__),SCRIPT,R/'4090-matchbox-long-training-plan-v1.json',R/'4090-matchbox-long-recovery-plan-v1.json']+[x for x in OLD.iterdir() if x.is_file()]
 p=dict(schema='matchbox-long-control-resume-plan-v1',files={str(p):sha(p) for p in paths},checkpoint_step=3500,original_complete_saved_batches=len(records),original_gzip_complete=dec.eof,remaining_updates=1500,output=str(NEW),policy='Verified process absent after infrastructure restart. Restore originalmodel/Adam/RNG states at3500, restore every sampler RNG and progress/source hashes. Resume3501..5000 only, no seed/recipe/checkpointselection or recognition rescue. Preserve old bytes and retain exact3500prefix metadata in new complete gzip; suffix aftercheckpoint remains only in interrupted original. Restore originalpreupdate representatives,initialweights/frontend. New finalreport uses resumed path; CPU checkpointdeserialization only, neuralupdates RTX4090. Additional atomic recovery checkpoint every500. No protected sources or neuralnumericalcomparison gate.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),step=d['step'],remaining=1500)))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for name,h in p['files'].items():assert sha(name)==h,name
 source=SCRIPT.read_text().replace("OUT.mkdir(exist_ok=False)","OUT.mkdir(exist_ok=False)")
 source=source.replace("f'matchbox-long-{ARM}-v1'","f'matchbox-long-{ARM}-resumed-v1'").replace("f'4090-matchbox-long-{ARM}-training-original-v1.json'","f'4090-matchbox-long-{ARM}-training-recovery-v1.json'")
 needle="  first=sampler.batch();start=time.monotonic();progress=[]"
 assert source.count(needle)==1
 hook='''  checkpoint=torch.load(RESUME_OLD/'recovery.pt',map_location='cpu',weights_only=False)
  assert checkpoint['step']==3500 and checkpoint['arm']==ARM
  model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
  assert all(int(v['step'])==3500 for v in optimizer.state.values())
  for e,state in checkpoint['sampler_rngs'].items():sampler.sources[e].bit_generator.state=state
  for n,state in checkpoint['sampler_other_rngs'].items():getattr(sampler,n).bit_generator.state=state
  sampler.decoded=checkpoint['decoded'];progress=checkpoint['progress'];start=time.monotonic()
  for name in ['initial.pt','frontend.pt','representative.npy','first-labels.npy']:shutil.copyfile(RESUME_OLD/name,OUT/name)
  dec=zlib.decompressobj(16+zlib.MAX_WBITS);prefix=dec.decompress((RESUME_OLD/'batches.jsonl.gz').read_bytes()).splitlines()[:3500]
  assert len(prefix)==3500
  with gzip.open(OUT/'batches.jsonl.gz','wt') as prefix_log:
   for line in prefix:prefix_log.write(line.decode()+'\\n')
  torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
  first=sampler.batch()
'''
 source=source.replace(needle,hook).replace("gzip.open(OUT/'batches.jsonl.gz','wt')","gzip.open(OUT/'batches.jsonl.gz','at')")
 source=source.replace("range(1,p['steps']+1)","range(3501,p['steps']+1)").replace('first if step==1 else','first if step==3501 else')
 source=source.replace("row=dict(step=step,loss=","row=dict(process_io=Path('/proc/self/io').read_text(),step=step,loss=")
 source=source.replace('    if step%500==0:disk()', '''    if step%500==0:
     disk()
     state=dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler_rngs={e:g.bit_generator.state for e,g in sampler.sources.items()},sampler_other_rngs={n:getattr(sampler,n).bit_generator.state for n in ['offsets','bg_rng','ambient_rng','permutation']},decoded=sampler.decoded,progress=progress,arm=ARM,training_plan_sha256=sha(LOCK),resume_plan_sha256=RESUME_SHA)
     tmp=OUT/'recovery.tmp';torch.save(state,tmp)
     with tmp.open('rb') as f:os.fsync(f.fileno())
     os.replace(tmp,OUT/'recovery.pt')''')
 source=source.replace("elapsed_seconds=time.monotonic()-start,all_training_checks_passed", "elapsed_seconds=time.monotonic()-start,resumed_from_step=3500,resume_plan_sha256=RESUME_SHA,all_training_checks_passed")
 sys.argv=[str(SCRIPT),'run','--arm','control']
 exec(compile(source,str(SCRIPT),'exec'),dict(__name__='__main__',__file__=str(SCRIPT),RESUME_OLD=OLD,RESUME_SHA=sha(LOCK),shutil=shutil,zlib=zlib))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
