"""Fixed training rendered-source diagnostic; no optimizer."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-multiscale-rendered-support-plan-v1.json';B=Path('.local/wakeword/multiscale-rendered-support-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plan():
 sourceplan=R/'4090-multiscale-factorial-control-plan-v1.json';ids={s['id'] for s in read(sourceplan)['positives']};ledger=Path('.local/wakeword/causal-onset-material-v1/records.jsonl');rows=[json.loads(x) for x in ledger.read_text().splitlines()];rows=[r for r in rows if r['source']['id'] in ids];assert len(rows)==1080
 train=R/'4090-binary40-multiscale-training-plan-v1.json';files=dict(read(train)['files'])
 paths=[Path(__file__),sourceplan,ledger,train,R/'4090-causal-command-prefix-reference-map-v1.json',Path('research/wakeword_p4/experiments/multiscale_rendered_support_protocol_v1.md')]
 paths += [Path(f'.local/wakeword/binary40-multiscale-training-v1/{a}/final.pt') for a in ['control','multiscale']]
 paths += [Path(p) for p in {r['feature_binary'] for r in rows}]
 for p in paths:files[str(p)]=sha(p)
 with LOCK.open('x') as f:json.dump({'schema':'multiscale-rendered-support-plan-v1','rows':rows,'files':files,'arms':['control','multiscale'],'optimizer_updates':0},f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=read(LOCK)
 for f,h in p['files'].items():assert sha(f)==h,f
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_span_v1 import Binary40Span,CONTROL
 from causal_binary40_multiscale_v1 import Binary40Multiscale
 from causal_sequence_supervision_v1 import masks
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name();B.mkdir(exist_ok=False)
 refs={r['id']:r for r in read(R/'4090-causal-command-prefix-reference-map-v1.json')['records']};bins={name:np.memmap(name,dtype='<u2',mode='r').reshape(-1,40) for name in {r['feature_binary'] for r in p['rows']}};inputs=[];positive=[];negative=[];metadata=[]
 for row in p['rows']:
  frames=row['frames'];assert frames<=846 and row['source']['label']==1
  raw=bins[row['feature_binary']][row['offset']:row['offset']+frames];x=np.zeros((846,40),np.float32);x[:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
  end=row['container_end_samples'];source=row['source'];ref=refs.get(source['id'])
  if ref:
   assert ref['native_sha256']==source['native_sha256'];tail=end-row['lead_samples']-source['native_samples'];assert tail>=0;end=row['lead_samples']+ref['automatic_prefix_end_sample']+tail
  _,pw,nw=masks(int(frames),1,int(row['lead_samples']),int(end));pp=np.zeros(282,np.float32);nn=np.zeros(282,np.float32);pp[:len(pw)]=pw;nn[:len(nw)]=nw
  inputs.append(x);positive.append(pp);negative.append(nn);metadata.append({'order':row['order'],'source_id':source['id'],'text':source['text'],'variant':row['variant'],'lead_samples':row['lead_samples'],'frames':frames,'reference_end':end})
 x=np.stack(inputs);pw=np.stack(positive);nw=np.stack(negative);np.savez_compressed(B/'inputs.npz',features=x,positive=pw,negative=nw)
 reports={}
 for arm in p['arms']:
  checkpoint=Path(f'.local/wakeword/binary40-multiscale-training-v1/{arm}/final.pt');cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==20000
  model=(Binary40Span(CONTROL) if arm=='control' else Binary40Multiscale()).cuda().eval();model.load_state_dict(cp['model'],strict=True);assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items());values=[]
  with torch.inference_mode():
   for i in range(0,len(x),32):
    batch=torch.from_numpy(x[i:i+32]).permute(0,2,1).cuda();z,_=model(batch,model.initial_state(len(batch),'cuda'));values.append(torch.sigmoid(z[:,0]).cpu().numpy())
  prob=np.concatenate(values);assert prob.shape==(1080,282) and np.isfinite(prob).all();trace=B/f'{arm}.npz';np.savez_compressed(trace,probability=prob)
  rows=[]
  for i,m in enumerate(metadata):
   avg=np.lib.stride_tricks.sliding_window_view(prob[i].astype(np.float64),5).mean(axis=1);mask=pw[i,4:]>0;assert mask.any();rows.append(dict(m,maximum_five_mean=float(avg[mask].max()),crossing=bool((avg[mask]>=.91796875).any()),positive_mean=float((prob[i]*pw[i]).sum())))
  reports[arm]={'rows':rows,'checkpoint_sha256':sha(checkpoint),'trace':{'path':str(trace),'sha256':sha(trace)},'crossings':sum(r['crossing'] for r in rows)};print(arm,reports[arm]['crossings'],flush=True)
 out={'schema':'multiscale-rendered-support-original-v1','plan_sha256':sha(LOCK),'inputs':{'path':str(B/'inputs.npz'),'sha256':sha(B/'inputs.npz')},'arms':reports,'optimizer_updates':0,'gpu':torch.cuda.get_device_name(),'limits':'Original selected trainingrenderings, not unseenvoice/acceptance/P4. Window support follows mechanical traininglabels, not phonetic truth; crossing is not a full decoder event.'}
 (R/'4090-multiscale-rendered-support-original-v1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
