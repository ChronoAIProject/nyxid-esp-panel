"""Independent coordinate-to-tensor audit of saved census; no sampler or model."""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=R/'4090-hey-broad-surviving-window-census-v2.json'
    if not p.exists():print(json.dumps(dict(status='not_ready',missing=str(p))));return 2
    report=json.loads(p.read_text());execution=R/'4090-hey-broad-surviving-window-census-execution-v2.json'
    assert report['execution_sha256']==sha(execution)
    lock=json.loads(execution.read_text())
    for path,h in lock['files'].items():assert sha(path)==h
    output=[]
    for target_report in report['targets']:
        target=target_report['target'];root=Path('.local/wakeword/hey-broad-training-features-v1')/target
        rows=json.loads((root/'features.json').read_text());raw=np.memmap(root/'features.bin',mode='r',dtype='<u2').reshape(-1,40)
        backgrounds=sum([json.loads((root/n).read_text()) for n in ('human.json','ambient.json')],[])
        arrays=[np.load(root/r['path'],mmap_mode='r') for r in backgrounds]
        counts=Counter();sources={};conditioning={};coverage=Counter();retained=discarded=0
        trace=Path(target_report['trace_path']);assert sha(trace)==target_report['trace_sha256']
        schedule=Path('.local/wakeword/hey-broad-schedules-v1')/(target+'.jsonl')
        with gzip.open(trace,'rt') as f,schedule.open() as s:
            for number,line in enumerate(f,1):
                batch=json.loads(line);wanted=json.loads(next(s));assert batch['step']==wanted['index']==number
                assert len(batch['survivors'])==128 and len(batch['overwritten_ordinary'])==16
                xs=[];ys=[];ws=[];slot_counts=Counter()
                for rec in batch['overwritten_ordinary']:
                    assert rec['label']==0 and rec['kind']=='ordinary' and rows[rec['row']]['label']==0
                    discarded+=1
                for rec in batch['survivors']:
                    start,end=rec['start'],rec['end'];assert end-start==225 and start>=0
                    if rec['kind']=='ambient':
                        data=arrays[rec['ambient_index']][start:end];assert rec['label']==0
                        key=('ambient','background','ambient','none','0','')
                    else:
                        row=rows[rec['row']];assert rec['offset']==row['offset'] and end<=row['frames']
                        if rec['kind']=='prefix':
                            assert rec['label']==0 and row['label']==1 and row['engine'] in ('piper','kokoro')
                            assert 15<=row['keyword_frame']-end<=35
                        else:
                            assert rec['kind']=='ordinary' and rec['label']==row['label']
                            if rec['label']:assert 3<=end-row['keyword_frame']<=27
                            else:assert max(225,row['lead_frame']-25)<=end<=row['frames']
                        offset=row['offset'];data=raw[offset+start:offset+end]
                        role='own' if row['target_id']==target else 'common' if row['target_id']=='common_negative' else 'reciprocal'
                        key=(rec['kind'],role,row['engine'],row['source_pair_key']['carrier'],str(rec['label']),row['text'])
                        sources.setdefault(key,set()).add(row['source_id']);conditioning.setdefault(key,set()).add(str(row['source_pair_key']['conditioning']))
                        coverage[rec['kind'],'windows']+=1
                        if row['keyword_frame'] is not None:coverage[rec['kind'],'contains_generated_prefix_boundary']+=int(start<=row['keyword_frame']<end)
                    assert data.shape==(225,40)
                    xs.append(np.clip(np.asarray(data,dtype=np.float32)/25.6,0,26));ys.append([rec['label']]);ws.append(1 if rec['label'] else 20)
                    slot_counts[rec['kind'],rec['label']]+=1;counts[key]+=1;retained+=1
                assert slot_counts=={('ordinary',1):32,('ordinary',0):32,('prefix',0):16,('ambient',0):48}
                h=hashlib.sha256()
                for a in (np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32),np.asarray(ws,dtype=np.float32)):h.update(a.tobytes())
                assert h.hexdigest()==batch['batch_sha256']==wanted['tensor_sha256']
                if number%1000==0:print(json.dumps(dict(target=target,batch=number)),flush=True)
        assert number==lock['batches_per_target']==target_report['batches']
        expected=[dict(slot=k[0],role=k[1],engine=k[2],carrier=k[3],label=int(k[4]),text=k[5],conditioning_ids=len(conditioning.get(k,set())),windows=v,unique_source_containers=len(sources.get(k,set()))) for k,v in sorted(counts.items())]
        assert expected==target_report['counts']
        assert [dict(slot=k[0],measure=k[1],count=v) for k,v in sorted(coverage.items())]==target_report['generated_boundary_coverage']
        output.append(dict(target=target,batches=number,retained_windows=retained,discarded_ordinary_windows=discarded,all_reconstructed_batches_match_original_schedule=True,aggregate_counts_exact=True,trace_sha256=sha(trace)))
    result=dict(all_passed=True,original_result_sha256=sha(p),source_sha256=sha(__file__),targets=output,model_invocations=0,limits='Independent saved-coordinate reconstruction, not original historical live provenance or phonetic/recognition/FAR evidence. First10000 batches only.')
    with (R/'4090-hey-broad-surviving-window-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
