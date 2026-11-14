"""Replay original CPU sampler with line observers; preserve RNG and batch bytes.

Coordinates are reconstructed observations, not a historical live recorder.
No model, optimizer, waveform decode or development/acceptance access.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import numpy as np
from schedule_hey_broad_v1 import sampler

R=Path('research/wakeword_p4/results')
PLAN=R/'4090-hey-broad-surviving-window-census-execution-v2.json'
OUT=Path('.local/wakeword/hey-broad-surviving-window-census-v2')
TRAINER=Path('research/wakeword_p4/training/train_stream.py')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def batch_sha(values):
    h=hashlib.sha256()
    for a in values: h.update(a.tobytes())
    return h.hexdigest()

def plan():
    paths=[Path(__file__),TRAINER,Path(__file__).with_name('schedule_hey_broad_v1.py'),Path(__file__).with_name('audit_hey_broad_batches_v1.py')]
    paths += [Path('research/wakeword_p4/training')/n for n in ('negative_sampling.py','spec_augment.py','ambient_sampling.py')]
    for target in ('hey_willow','hey_nyx'):
        paths.append(Path('.local/wakeword/hey-broad-schedules-v1')/(target+'.jsonl'))
        root=Path('.local/wakeword/hey-broad-training-features-v1')/target
        paths += [root/n for n in ('features.json','target-provenance.json','human.json','ambient.json')]
    p=dict(schema='hey-broad-surviving-window-census-execution-v1',files={str(p):sha(p) for p in paths},batches_per_target=10000,targets=['hey_willow','hey_nyx'],
           policy='Observe original extracted sample/prefix/batch line events without RNG calls or source mutation; first2 traced-vs-original exact; every complete128 batch must match original schedule. Log original48 ordinary draws and16 overwritten plus actual128 shuffled survivors; every coordinate scaled slice must equal corresponding actual network row. Aggregate actual surviving word/source/carrier/label plus generated-boundary coverage. CPU only, no models/audio/newtraining/freshdata. First10000 only; not all120000 or historical live provenance, not phonetic truth.')
    with PLAN.open('x') as f: f.write(json.dumps(p,indent=2)+'\n')

class Observer:
    def __init__(self,env,target):
        self.env=env;self.target=target;self.draws=[];self.prefix=[];self.noise=[];self.order=None
        self.lines={}
        for number,line in enumerate(TRAINER.read_text().splitlines(),1):
            for key,text in [('sample','output.append(data[end-frames:end])'),('prefix_sample','output.append(sequence(row)[end-frames:end])'),('batch','noise.append(data[at:at+frames])')]:
                if line.strip()==text: self.lines[key]=number
        assert len(self.lines)==3
        self.codes={env[n].__code__:n for n in ('sample','prefix_sample','batch')}
        rows=json.loads((Path('.local/wakeword/hey-broad-training-features-v1')/target/'features.json').read_text())
        self.rows={r['offset']:r for r in rows};self.row_index={r['offset']:i for i,r in enumerate(rows)}
    def callback(self,frame,event,arg):
        name=self.codes.get(frame.f_code)
        if name is None:return None
        if event=='line' and frame.f_lineno==self.lines[name]:
            d=frame.f_locals
            if name=='batch': self.noise.append(dict(kind='ambient',ambient_index=int(d['index']),start=int(d['at']),end=int(d['at'])+225,label=0))
            else:
                row=d['row'];end=int(d['end']);rec=dict(kind='prefix' if name=='prefix_sample' else 'ordinary',row=self.row_index[row['offset']],offset=row['offset'],end=end,start=end-225,label=int(d.get('label',0)))
                (self.prefix if name=='prefix_sample' else self.draws).append(rec)
        if name=='batch' and event=='return':self.order=frame.f_locals['order'].copy()
        return self.callback
    def batch(self):
        self.draws=[];self.prefix=[];self.noise=[];self.order=None
        assert sys.gettrace() is None
        sys.settrace(self.callback)
        try: values=self.env['batch'](mine=False,augment=False)
        finally:sys.settrace(None)
        assert len(self.draws)==80 and len(self.prefix)==16 and len(self.noise)==48
        assert [r['label'] for r in self.draws]==[1]*32+[0]*48
        overwritten=self.draws[32:48]
        original=self.draws[:32]+self.prefix+self.draws[48:]+self.noise
        survivors=[original[int(i)] for i in self.order]
        x,y,w=values
        for i,rec in enumerate(survivors):
            if rec['kind']=='ambient': data=self.env['ambient']['training'][rec['ambient_index']][1][rec['start']:rec['end']]
            else:
                at=rec['offset'];data=self.env['raw'][at+rec['start']:at+rec['end']]
            expected=np.clip(np.asarray(data,dtype=np.float32)/25.6,0,26)
            assert np.array_equal(x[i],expected) and y[i,0]==rec['label'] and w[i]==(1 if rec['label'] else 20)
        return values,survivors,overwritten

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
    p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    OUT.mkdir(exist_ok=False);reports=[]
    for target in p['targets']:
        # Fresh original sampler generates first2 without observers; exact tensors
        # are compared to reconstructed coordinates and observed full batches.
        untraced=sampler(target);first=[tuple(a.copy() for a in untraced['batch'](mine=False,augment=False)) for _ in range(2)];del untraced
        env=sampler(target);observer=Observer(env,target)
        schedule=Path('.local/wakeword/hey-broad-schedules-v1')/(target+'.jsonl')
        counts=Counter();coverage=Counter();sources={};conditioning={};trace=OUT/(target+'.jsonl.gz')
        with schedule.open() as expected, gzip.open(trace,'xt') as ledger:
            for step in range(1,p['batches_per_target']+1):
                values,survivors,overwritten=observer.batch()
                wanted=json.loads(next(expected));digest=batch_sha(values)
                assert wanted['index']==step and wanted['tensor_sha256']==digest
                if step<=2:
                    for a,b in zip(values,first[step-1],strict=True):assert a.tobytes()==b.tobytes()
                for rec in survivors:
                    if rec['kind']=='ambient':key=('ambient','background','ambient','none','0','')
                    else:
                        row=observer.rows[rec['offset']];role='own' if row['target_id']==target else 'common' if row['target_id']=='common_negative' else 'reciprocal'
                        key=(rec['kind'],role,row['engine'],row['source_pair_key']['carrier'],str(rec['label']),row['text'])
                        sources.setdefault(key,set()).add(row['source_id'])
                        conditioning.setdefault(key,set()).add(str(row['source_pair_key']['conditioning']))
                        if row['keyword_frame'] is not None:
                            coverage[(rec['kind'],'contains_generated_prefix_boundary')]+=int(rec['start']<=row['keyword_frame']<rec['end'])
                        coverage[(rec['kind'],'windows')]+=1
                    counts[key]+=1
                ledger.write(json.dumps(dict(step=step,batch_sha256=digest,survivors=survivors,overwritten_ordinary=overwritten),separators=(',',':'))+'\n')
                if step%100==0:
                    ledger.flush();print(json.dumps(dict(target=target,step=step,all_batch_hashes_match=True)),flush=True)
        reports.append(dict(target=target,batches=p['batches_per_target'],first2_traced_untraced_exact=True,all_batch_hashes_match=True,every_surviving_coordinate_matches_network_input=True,trace_path=str(trace),trace_sha256=sha(trace),
                            counts=[dict(slot=k[0],role=k[1],engine=k[2],carrier=k[3],label=int(k[4]),text=k[5],conditioning_ids=len(conditioning.get(k,set())),windows=v,unique_source_containers=len(sources.get(k,set()))) for k,v in sorted(counts.items())],generated_boundary_coverage=[dict(slot=k[0],measure=k[1],count=v) for k,v in sorted(coverage.items())]))
    result=dict(execution_sha256=sha(PLAN),targets=reports,model_invocations=0,limits=p['policy'])
    with (R/'4090-hey-broad-surviving-window-census-v2.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);args=ap.parse_args();plan() if args.action=='plan' else run()
