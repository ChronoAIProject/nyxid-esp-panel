"""One fixed5000update GPU run, new pretrained-window representation package."""
import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from matchbox_window_batch_v1 import Sampler,MATERIAL,SOURCE,digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-matchbox-window-training-plan-v1.json';OUT=Path('.local/wakeword/matchbox-window-v1')
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from matchboxnet_pretrained_v1 import ARCHIVE,ARCHIVE_SHA
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 material=json.loads(MATERIAL.read_text());assert material['synthetic_bare_presentations']==378 and material['training_background_original_utterances']==1004
 files=dict(material['files']);files[str(MATERIAL)]=sha(MATERIAL)
 for row in material['records']:
  assert row['role']=='training' and sha(row['path'])==row['sha256'];files[row['path']]=row['sha256']
 paths=[Path(__file__),Path(__file__).with_name('matchbox_window_batch_v1.py'),Path(__file__).with_name('matchbox_window_protocol_v1.md'),Path('research/wakeword_p4/tools/check_disk_budget.py'),ARCHIVE]
 files.update({str(p):sha(p) for p in paths});assert files[str(ARCHIVE)]==ARCHIVE_SHA
 sampler=Sampler(load_pcm=False)
 contracts=[]
 for index in range(4):
  d=sampler.descriptors();contracts.append(dict(batch=index,descriptors_sha256=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),class_counts=[sum(x['label']==i for x in d) for i in range(7)]))
 p=dict(schema='matchbox-window-training-plan-v1',files=files,versions={n:metadata.version(n) for n in ['torch','numpy','soundfile']},seed=726,steps=5000,batch=128,window_samples=20480,classes=7,optimizer=dict(lr=.0001,betas=[.9,.999],eps=1e-8,weight_decay=0,foreach=False,fused=False),frozen_bn_statistics=True,train_bn_affine=True,expected_new_gib=1,metadata_contract=contracts,representative='Actual first128GPU-MFCC windows before firstupdate plus literalzeroPCM-MFCC window; save exactfloat32[129,128,1,64] bytes. No retry.',policy='Exactly one final5000update RTX4090 FP32 new encoder/frontend/full-window task experiment under matchbox_window_protocol_v1.md. Not isolated pretraining attribution. All source/voice roles retain original limits. No Melo training, protected/acceptance access, seed/LR/checkpoint/head/calibration rescue, freshthreshold selection or model numerical comparison admission. No promotion; v30/Mac unchanged.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),steps=5000,source_files=len(files))),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 for package,v in p['versions'].items():assert metadata.version(package)==v
 def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
 disk();OUT.mkdir(exist_ok=False)
 launch=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch)+'\n')
 try:
  import torch
  from torch import nn
  from matchboxnet_pretrained_v1 import load
  assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
  torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
  front,model=load();model.decoder.decoder_layers=nn.Sequential(nn.Linear(128,7));front=front.cuda().eval();model=model.cuda().eval()
  parameters=list(model.parameters());assert sum(x.numel() for x in parameters)==74247 and len(parameters)==34 and all(x.requires_grad for x in parameters)
  def buffers():return {n:hashlib.sha256(b.detach().cpu().numpy().tobytes()).hexdigest() for n,b in model.named_buffers()}
  frozen=buffers();assert len(frozen)==27
  optimizer=torch.optim.Adam(parameters,**p['optimizer']);assert len(optimizer.state)==0
  torch.save(model.state_dict(),OUT/'initial.pt');torch.save(front.state_dict(),OUT/'frontend.pt')
  sampler=Sampler(p['seed']);(OUT/'decoded-source-hashes.json').write_text(json.dumps(sampler.decoded,indent=2)+'\n')
  first=sampler.batch();start=time.monotonic();progress=[]
  with gzip.open(OUT/'batches.jsonl.gz','wt') as log:
   for step in range(1,p['steps']+1):
    pcm,labels,descriptors=first if step==1 else sampler.batch()
    wave=torch.from_numpy(pcm.astype(np.float32)/32768).cuda();target=torch.from_numpy(labels).cuda()
    with torch.no_grad():features=front(wave)
    assert features.device.type=='cuda' and features.shape==(128,64,128)
    if step==1:
     with torch.no_grad():silence=front(torch.zeros(1,20480,device='cuda'))
     rep=torch.cat((features,silence)).permute(0,2,1).unsqueeze(2).cpu().numpy();assert rep.shape==(129,128,1,64) and np.isfinite(rep).all();np.save(OUT/'representative.npy',rep)
     np.save(OUT/'first-labels.npy',labels)
    optimizer.zero_grad(set_to_none=True);logits=model(features);loss=nn.functional.cross_entropy(logits,target);loss.backward()
    assert logits.device.type=='cuda' and logits.shape==(128,7)
    assert all(x.grad is not None for x in parameters)
    assert bool(torch.stack([torch.isfinite(loss)]+[torch.isfinite(x.grad).all() for x in parameters]).all())
    optimizer.step()
    log.write(json.dumps(dict(step=step,pcm_sha256=digest(pcm),labels_sha256=digest(labels),features_sha256=digest(features.detach().cpu().numpy()),descriptors=descriptors))+'\n')
    if step==1 or step%100==0:
     row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-start,device=str(features.device),peak_allocated_bytes=torch.cuda.max_memory_allocated(),frozen_bn_buffers_unchanged=buffers()==frozen);assert row['frozen_bn_buffers_unchanged'];progress.append(row);print(json.dumps(row),flush=True);log.flush()
    if step%500==0:disk()
  assert buffers()==frozen and all(int(v['step'])==5000 for v in optimizer.state.values()) and len(optimizer.state)==34
  torch.save(model.state_dict(),OUT/'final.pt');torch.save(optimizer.state_dict(),OUT/'optimizer-final.pt')
  paths=[x for x in OUT.iterdir() if x.is_file()]
  report=dict(schema='matchbox-window-training-original-v1',plan_sha256=sha(LOCK),launch=launch,steps=5000,optimizer_iterations=5000,device=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,all_training_checks_passed=True,frozen_bn_buffer_hashes=frozen,progress=progress,files={x.name:dict(bytes=x.stat().st_size,sha256=sha(x)) for x in paths},recognition_scored=False,acceptance_accessed=False,promotion_qualified=False,policy=p['policy'])
  with (R/'4090-matchbox-window-training-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
  print(json.dumps(dict(completed=True,steps=5000,elapsed_seconds=report['elapsed_seconds'])),flush=True)
 except BaseException:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
