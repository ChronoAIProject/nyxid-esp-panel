"""Infrastructure recovery only; preserve interrupted original and fixed recipe."""
import argparse,hashlib,json,os,subprocess,sys,zlib
from pathlib import Path
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword');SCRIPT=Path('research/wakeword_p4/experiments/train_matchbox_long_v1.py');LOCK=R/'4090-matchbox-long-recovery-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 old=ROOT/'matchbox-long-control-v1';launch=json.loads((old/'launch.json').read_text());assert not Path(f"/proc/{launch['pid']}").exists()
 assert not (old/'final.pt').exists() and not (old/'optimizer-final.pt').exists()
 raw=(old/'batches.jsonl.gz').read_bytes();decoder=zlib.decompressobj(16+zlib.MAX_WBITS);decoded=decoder.decompress(raw);records=[]
 for line in decoded.splitlines():
  try:records.append(json.loads(line))
  except json.JSONDecodeError:pass
 assert [r['step'] for r in records]==list(range(1,len(records)+1))
 report=dict(schema='matchbox-long-original-infrastructure-interruption-v1',launch=launch,process_absent=True,original_session_handle_missing=True,complete_batch_records=len(records),last_complete_batch_record=records[-1]['step'],gzip_footer_complete=decoder.eof,final_weights_available=False,optimizer_available=False,exact_resume_possible=False,files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in old.iterdir() if p.is_file()},interpretation='Original trace incomplete due host/daemon restart. Last saved batch is not necessarily last executed optimizer update. No recognition outcome or selected checkpoint; no original bytes modified.')
 receipt=R/'4090-matchbox-long-original-interruption-v1.json'
 with receipt.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 paths=[Path(__file__),SCRIPT,receipt,R/'4090-matchbox-long-training-plan-v1.json']
 p=dict(schema='matchbox-long-recovery-plan-v1',files={str(p):sha(p) for p in paths},arms=['control','pooled'],steps_each=5000,seed=726,checkpoint_every=500,policy='One new full run per arm because original exact resume state is missing after verified infrastructure interruption; not performance rescue. Original5000step recipe/seed/optimizer/source sampler/frontend/finalonly selection unchanged. Original interrupted outputs retained byte-for-byte. Separate recovered output/report names. Atomic overwritten recovery checkpoint every500: model,optimizer,torch/CUDA and sampler RNGs,progress/sourcehashes. Checkpoints for infrastructure recovery only, never recognition/selection. No neural numerical-equality gate, new source or heldout consumption. Record /proc/self/io every100steps. Execution wrapper uses bound original runner and applies only explicit logging/output/checkpoint hooks.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),interrupted_complete_records=len(records))))
def run(arm):
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 source=SCRIPT.read_text()
 source=source.replace("f'matchbox-long-{ARM}-v1'","f'matchbox-long-{ARM}-recovery-v1'")
 source=source.replace("f'4090-matchbox-long-{ARM}-training-original-v1.json'","f'4090-matchbox-long-{ARM}-training-recovery-v1.json'")
 source=source.replace("row=dict(step=step,loss=", "row=dict(process_io=Path('/proc/self/io').read_text(),step=step,loss=")
 needle='    if step%500==0:disk()'
 assert source.count(needle)==1
 replacement='''    if step%500==0:
     disk()
     payload=dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler_rngs={e:g.bit_generator.state for e,g in sampler.sources.items()},sampler_other_rngs={n:getattr(sampler,n).bit_generator.state for n in ['offsets','bg_rng','ambient_rng','permutation']},decoded=sampler.decoded,progress=progress,arm=ARM,training_plan_sha256=sha(LOCK),recovery_plan_sha256=RECOVERY_SHA)
     tmp=OUT/'recovery.tmp';torch.save(payload,tmp)
     with tmp.open('rb') as checkpoint_file:os.fsync(checkpoint_file.fileno())
     os.replace(tmp,OUT/'recovery.pt')
'''
 source=source.replace(needle,replacement)
 sys.argv=[str(SCRIPT),'run','--arm',arm]
 exec(compile(source,str(SCRIPT),'exec'),dict(__name__='__main__',__file__=str(SCRIPT),RECOVERY_SHA=sha(LOCK)))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);parser.add_argument('--arm',choices=['control','pooled']);a=parser.parse_args()
 if a.action=='plan':plan()
 else:
  assert a.arm;run(a.arm)
