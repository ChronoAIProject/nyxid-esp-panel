"""Audit all saved formal states and reconstruct every recorded raw batch."""
import gzip
import h5py
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

R=Path('research/wakeword_p4/results')
BASE=Path('.local/wakeword/multispeaker-formal-v1')
FEATURES=Path('.local/wakeword/multispeaker-coverage-features-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def main():
    source=R/'4090-multispeaker-formal-original-v1.json'
    report=json.loads(source.read_text());lock=R/'4090-multispeaker-formal-execution-v1.json'
    plan=json.loads(lock.read_text());assert report['execution_sha256']==sha(lock)
    assert report['formal_training_completed'] and not report['development_scored'] and not report['acceptance_scored']
    inventory=plan['initial']['variable_inventory']['variables'];arms=[]
    # Keras H5 numbers the four unwrapped conv layers separately from the
    # first stream-wrapped conv. Bind by architecture/name, not shape alone.
    h5_mapping={}
    for e in inventory:
        name=e['name']
        if name=='stream/conv2d/kernel':path='layers/stream/cell/vars/0'
        else:
            layer,variable=name.rsplit('/',1)
            if layer.startswith('conv2d_'):
                index=int(layer.split('_')[-1])-1
                layer='conv2d'+('_'+str(index) if index else '')
            slot={'kernel':0,'bias':1,'gamma':0,'beta':1,'moving_mean':2,'moving_variance':3}[variable]
            path=f'layers/{layer}/vars/{slot}'
        h5_mapping[e['key']]=path
    assert len(set(h5_mapping.values()))==31
    with h5py.File(plan['initial']['snapshot']['path'],'r') as h5:
        for e in inventory:assert digest(h5[h5_mapping[e['key']]][()])==e['sha256']
    for item in report['arms']:
        arm=item['arm'];root=BASE/arm
        assert item['steps']==item['optimizer_iterations']==20000
        for name,h in item['files'].items():assert sha(root/name)==h,name
        changed=[]
        with np.load(root/'initial-model-variables.npz',allow_pickle=False) as a,np.load(root/'final-model-variables.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)=={e['key'] for e in inventory}
            for e in inventory:
                before,after=a[e['key']],b[e['key']]
                assert digest(before)==e['sha256'] and before.shape==after.shape and before.dtype==after.dtype
                assert np.isfinite(after).all()
                if e['effective_trainable']:
                    if before.tobytes()!=after.tobytes():changed.append(e['index'])
                else:assert before.tobytes()==after.tobytes()
        assert len(changed)==15
        with h5py.File(root/'final.weights.h5','r') as h5,np.load(root/'final-model-variables.npz',allow_pickle=False) as saved:
            for key,path in h5_mapping.items():
                a,b=h5[path][()],saved[key]
                assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        with np.load(root/'initial-optimizer-variables.npz',allow_pickle=False) as a,np.load(root/'final-optimizer-variables.npz',allow_pickle=False) as b:
            assert len(a.files)==len(b.files)==32
            assert a['v000'].item()==0 and b['v000'].item()==20000
            assert a['v001'].item()==b['v001'].item()==np.float32(1e-5)
            assert all(np.all(a[f'v{i:03}']==0) for i in range(2,32))
            assert all(np.isfinite(b[k]).all() for k in b.files)
            assert all(np.any(b[f'v{i:03}']!=0) for i in range(2,32))
        arms.append(dict(arm=arm,all31_initial_variables_exact=True,final_h5_matches_all31_saved_arrays=True,frozen16_unchanged=True,changed_trainable_indices=changed,optimizer_initial_zero_slots=30,optimizer_final_iteration=20000))
    assert {a['arm'] for a in arms}=={'narrow','broad'}
    shared=np.memmap(FEATURES/'shared-features.bin',dtype='<u2',mode='r').reshape(-1,40)
    rows={r['shared_index']:r for r in json.loads((FEATURES/'shared-metadata.json').read_text())}
    ambient=sum([json.loads((FEATURES/'narrow'/n).read_text()) for n in ['human.json','ambient.json']],[])
    arrays={}
    def reconstruct(arm,record):
        descriptors=record['descriptors'];assert len(descriptors)==128
        windows=[];labels=[];counts={'positive':0,'negative':0,'ambient':0}
        for d in descriptors:
            end=d['end'];assert end>=309
            if d['kind']=='synthetic':
                row=rows[d['index']]
                assert row['split']=='training' and arm in row['arms']
                assert d['label']==row['label']==int(row['target_id']=='hey_nyx')
                assert d['cell']==[row['engine'],row['label'],row['text'],row['source_pair_key']['scale'],row['variant']]
                assert end<=row['frames']
                if row['label']:
                    assert 3<=end-row['keyword_frame']<=27 and end-309<=row['lead_frame']
                    counts['positive']+=1
                else:
                    assert end>=max(309,row['lead_frame']-25);counts['negative']+=1
                start=row['offset']+end-309;window=shared[start:start+309]
            else:
                assert d['kind']=='ambient' and d['label']==0
                row=ambient[d['index']];assert row['split']=='training'
                if d['index'] not in arrays:
                    path=FEATURES/arm/row['path'];assert sha(path)==plan['files'][str(path)]
                    arrays[d['index']]=np.load(path,mmap_mode='r',allow_pickle=False)
                values=arrays[d['index']];assert end<=len(values)
                window=values[end-309:end];counts['ambient']+=1
            assert window.shape==(309,40);windows.append(window);labels.append(d['label'])
        assert counts==dict(positive=32,negative=48,ambient=48)
        x=np.asarray(windows,dtype=np.float32);x/=np.float32(25.6);np.clip(x,0,26,out=x)
        y=np.asarray(labels,dtype=np.float32).reshape(128,1)
        w=np.asarray([1. if v else 20. for v in labels],dtype=np.float32)
        for name,a in [('x',x),('y',y),('weights',w)]:assert digest(a)==record[name+'_sha256']
        assert np.isfinite(record['loss'])
    count=0
    with gzip.open(BASE/'narrow/ordered-batches.jsonl.gz','rt') as na,gzip.open(BASE/'broad/ordered-batches.jsonl.gz','rt') as br:
        for step,pair in enumerate(itertools.zip_longest(na,br),1):
            assert None not in pair
            a,b=map(json.loads,pair);assert a['step']==b['step']==step
            reconstruct('narrow',a);reconstruct('broad',b)
            for da,db in zip(a['descriptors'],b['descriptors'],strict=True):
                assert da['kind']==db['kind'] and da['label']==db['label']
                if da['kind']=='ambient':assert da==db
                else:assert da['cell']==db['cell']
            count=step
            if step%1000==0:print(json.dumps(dict(paired_steps_verified=step)),flush=True)
    assert count==20000
    result=dict(schema='multispeaker-formal-independent-audit-v1',auditor_sha256=sha(__file__),original_report_sha256=sha(source),execution_sha256=sha(lock),arms=arms,batches_reconstructed=40000,raw_windows_reconstructed=5120000,all_batch_digests_labels_roles_geometry_and_paired_cells_match=True,all_saved_state_checks_passed=True,model_calls=0,limits='All recorded batches reconstructed directly from raw inputs; saved initial/final arrays and artifact hashes verified. Does not independently reexecute gradients or quantify streaming INT8 accuracy or establish recognition/development/acceptance.')
    with (R/'4090-multispeaker-formal-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
