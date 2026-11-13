"""Whole binary/label/source audit and bounded independent frontend witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from pymicro_features import MicroFrontend

BASE=Path('.local/wakeword/hey-broad-training-features-v1');RESULTS=Path('research/wakeword_p4/results')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);p.add_argument('--starttime');a=p.parse_args()
    if a.wait_pid:
        assert a.starttime;proc=Path(f'/proc/{a.wait_pid}/stat')
        while proc.exists():
            try:s=proc.read_text().rsplit(')',1)[1].split()[19]
            except FileNotFoundError:break
            if s!=a.starttime:break
            time.sleep(5)
    rp=RESULTS/'4090-hey-broad-features-v1.json';d=json.loads(rp.read_text());pp=RESULTS/'4090-hey-broad-feature-plan-v1.json';plan=json.loads(pp.read_text())
    assert sha(pp)==d['plan_sha256'] and d['rows']==106956
    binary=BASE/'shared-features.bin';assert sha(binary)==d['shared_binary_sha256'] and binary.stat().st_size==d['shared_binary_bytes']==d['frames']*40*2
    raw=np.memmap(binary,dtype='<u2',mode='r').reshape(-1,40)
    reference=None;witnesses={};counts={};offset=0
    for target in ['hey_willow','hey_nyx']:
        root=BASE/target;prov=json.loads((root/'target-provenance.json').read_text())
        assert prov==d['targets'][target] and prov['target_id']==target and prov['schema']=='hey-broad-context225-target-pack-v1'
        for name,h in prov['files'].items():assert sha(root/name)==h
        assert (root/'features.bin').stat().st_ino==binary.stat().st_ino
        rows=json.loads((root/'features.json').read_text());assert len(rows)==106956
        if reference is None:reference=rows
        positives=0;end=0
        for i,r in enumerate(rows):
            assert r['split']=='training' and r['label']==int(r['target_id']==target)
            assert r['offset']==end and r['frames']>=225;end+=r['frames'];positives+=r['label']
            if r['keyword_frame'] is not None:assert r['keyword_frame']+27-225<=r['lead_frame'] and r['keyword_frame']+27<=r['frames']
            if target=='hey_nyx':assert {k:v for k,v in r.items() if k!='label'}=={k:v for k,v in reference[i].items() if k!='label'}
            else:
                key=(r['engine'],r['target_id'],r['variant']);witnesses.setdefault(key,[i,i])[1]=i
        assert end==len(raw) and positives==25128
        for name in ['human.json','ambient.json']:
            expected=[r for r in json.loads((Path('.local/wakeword/features-gpu-ami-extra-v1')/name).read_text()) if r['split']=='training']
            actual=json.loads((root/name).read_text());assert expected==actual
            for r in actual:
                p=root/r['path'];original=Path('.local/wakeword/features-gpu-ami-extra-v1')/r['path']
                assert sha(p)==sha(original) and np.load(p,mmap_mode='r').shape[0]>=225
        counts[target]=dict(positive=positives,negative=len(rows)-positives)
    # First+last in every(engine,target,variant) stratum, selected without scores.
    selected=sorted({i for pair in witnesses.values() for i in pair});receipts=[]
    for i in selected:
        r=reference[i];wav=Path(r['path']);assert sha(wav)==r['sha256'];pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and len(pcm)==r['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
        frontend=MicroFrontend();values=[]
        padded=np.pad(pcm,(0,(-len(pcm))%160))
        for start in range(0,len(padded),160):
            output=frontend.process_samples(padded[start:start+160].astype('<i2').tobytes());assert output.samples_read==160
            if output.features:values.append(np.rint(np.asarray(output.features)*25.6).astype('<u2'))
        expected=np.asarray(values,dtype='<u2').reshape(-1,40);actual=raw[r['offset']:r['offset']+r['frames']]
        assert np.array_equal(expected,actual)
        receipts.append(dict(index=i,engine=r['engine'],target=r['target_id'],variant=r['variant'],frames=len(expected),raw_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
    result=dict(schema='hey-broad-feature-independent-audit-v1',script_sha256=sha(Path(__file__)),original_report_sha256=sha(rp),
        rows=106956,frames=len(raw),binary_hash_verified=True,paired_all_labels_and_offsets_exact=True,training_only_ambient_exact=True,
        counts=counts,frontend_witnesses=receipts,
        limits='Whole byte/row/paired-role/ambient audit; independent frontend traversal only bounded first/last strata witnesses, same underlying pinned C++ frontend library. Not all106956 feature transforms independently recomputed, no model training/export/recognition evidence.')
    with (RESULTS/'4090-hey-broad-feature-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=106956,witnesses=len(receipts),all_checks_pass=True)),flush=True)
if __name__=='__main__':main()
