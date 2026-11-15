"""Pinned single intervention runner, original recipe plus grid sampler/checkpoints."""
import argparse,hashlib,json,subprocess
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-matchbox-grid-aligned-training-plan-v1.json';SCRIPT=Path('research/wakeword_p4/experiments/train_matchbox_long_v1.py')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def source():
 s=SCRIPT.read_text();old="from matchbox_long_batch_v1 import Sampler,MATERIAL,ROWS,digest";assert s.count(old)==1
 s=s.replace(old,"from matchbox_long_batch_v1 import MATERIAL,ROWS,digest\nfrom matchbox_grid_aligned_batch_v1 import Sampler")
 s=s.replace("LOCK=R/'4090-matchbox-long-training-plan-v1.json'","LOCK=R/'4090-matchbox-grid-aligned-training-plan-v1.json'")
 s=s.replace("f'4090-matchbox-long-{ARM}-training-original-v1.json'","'4090-matchbox-grid-aligned-training-original-v1.json'")
 s=s.replace("row=dict(step=step,loss=","row=dict(process_io=Path('/proc/self/io').read_text(),step=step,loss=")
 needle='    if step%500==0:disk()';assert s.count(needle)==1
 s=s.replace(needle,"""    if step%500==0:
     disk()
     payload=dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler_rngs={e:g.bit_generator.state for e,g in sampler.sources.items()},sampler_other_rngs={n:getattr(sampler,n).bit_generator.state for n in ['offsets','bg_rng','ambient_rng','permutation']},decoded=sampler.decoded,progress=progress,arm=ARM,training_plan_sha256=sha(LOCK))
     tmp=OUT/'recovery.tmp';torch.save(payload,tmp)
     with tmp.open('rb') as checkpoint_file:os.fsync(checkpoint_file.fileno())
     os.replace(tmp,OUT/'recovery.pt')
""")
 return s
def plan():
 parent=R/'4090-matchbox-long-training-plan-v1.json';p=json.loads(parent.read_text());paths=[Path(__file__),SCRIPT,parent,Path('research/wakeword_p4/experiments/matchbox_grid_aligned_batch_v1.py'),Path('research/wakeword_p4/experiments/matchbox_grid_alignment_protocol_v1.md'),R/'4090-matchbox-grid-aligned-batch-contract-v1.json',R/'4090-matchbox-mfcc-phase-audit-v1.json']
 p['files'].update({str(f):sha(f) for f in paths})
 for f,h in p['files'].items():assert sha(f)==h,f
 contract=json.loads((R/'4090-matchbox-grid-aligned-batch-contract-v1.json').read_text());assert contract['fullphrase_draws']==480000 and contract['all_original_source_class_pair_order_background_ambient_descriptors_identical']
 p.update(schema='matchbox-grid-aligned-training-plan-v1',arms=['pooled'],parent_plan_sha256=sha(parent),executable_source_sha256=hashlib.sha256(source().encode()).hexdigest(),checkpoint_every=500,policy=Path('research/wakeword_p4/experiments/matchbox_grid_alignment_protocol_v1.md').read_text())
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(sha(LOCK),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 s=source();assert hashlib.sha256(s.encode()).hexdigest()==p['executable_source_sha256']
 ns=dict(__name__='grid_aligned_runner',__file__=str(SCRIPT));exec(compile(s,str(SCRIPT),'exec'),ns);ns['ARM']='pooled';ns['OUT']=Path('.local/wakeword/matchbox-grid-aligned-v1');ns['run']()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
