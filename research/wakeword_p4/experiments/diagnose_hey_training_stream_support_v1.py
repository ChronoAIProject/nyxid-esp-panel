"""Locked full-training-only frozen-export support diagnostic; never fit/select."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import numpy as np
from uint8_stream_detector_v1 import Uint8StreamDetector
from diagnose_hey_consumed_raw_v1 import peaks

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-training-stream-diagnostic-plan-v1.json'
OUT=BASE/'hey-training-stream-diagnostic-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def versions():return {p:importlib.metadata.version(p) for p in ('numpy','ai-edge-litert')}

def register():
    lockp=RESULTS/'4090-hey-v2-formal-training-lock-v1.json';lock=json.loads(lockp.read_text())
    candidate=RESULTS/'4090-hey-v2-fixed-final-candidates-v1.json';models=json.loads(candidate.read_text())['models']
    files={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('uint8_stream_detector_v1.py'),Path(__file__).with_name('diagnose_hey_consumed_raw_v1.py'),lockp,candidate)}
    for target,arm in lock['arms'].items():
        root=Path(arm['features'])
        for name in ('features.bin','features.json'):
            p=root/name;assert sha(p)==lock['files'][str(p)];files[str(p)]=sha(p)
        assert sha(models[target]['path'])==models[target]['sha256'];files[models[target]['path']]=models[target]['sha256']
    import ai_edge_litert
    for p in Path(ai_edge_litert.__file__).parent.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.so'):files[str(p)]=sha(p)
    plan=dict(schema='hey-training-stream-diagnostic-plan-v1',files=files,models=models,arms=lock['arms'],versions=versions(),rows_per_arm=7200,positive_per_arm=720,negative_per_arm=6480,hypothesis='Test whether low development response also appears across complete training voices/environments, versus a training/development support gap. Full saved frontend training rows only; no new dev inference, fitting or threshold selection.',procedure='Fresh reference INT8 reset per complete training row; preserve every raw byte and peaks for all7200rows per arm, then source/label/condition summaries. No row filtering/retries.',acceptance_accessed=False,candidate_promoted=False,limits='Post-failure descriptive support diagnosis, not a new independent evaluation, identified cause or rescue.')
    with PLAN.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),inference_run=False)))

def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();p=json.loads(remote);assert versions()==p['versions']
    for name,h in p['files'].items():assert sha(name)==h
    OUT.mkdir(exist_ok=False);summaries={}
    for target in ('hey_willow','hey_nyx'):
        root=Path(p['arms'][target]['features']);rows=json.loads((root/'features.json').read_text());assert len(rows)==7200
        stored=np.memmap(root/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        detector=Uint8StreamDetector(p['models'][target]['path']);dest=OUT/target;dest.mkdir();records=[];offset=0
        with (dest/'raw-uint8.bin').open('xb') as rawfile,(dest/'records.jsonl').open('x') as log:
            for i,row in enumerate(rows):
                assert row['split']=='training'
                raw=stored[row['offset']:row['offset']+row['frames']]
                detector.reset();output=detector.predict(raw);assert output.dtype==np.uint8 and len(output)==row['frames']//3
                values=output.tobytes();rawfile.write(values)
                item=dict(index=i,record=row,trace_offset=offset,trace_bytes=len(values),trace_sha256=hashlib.sha256(values).hexdigest(),**peaks(output,row['samples']))
                records.append(item);log.write(json.dumps(item)+'\n');offset+=len(values)
                if (i+1)%100==0:print(json.dumps(dict(target=target,completed=i+1,planned=7200)),flush=True)
        summary=[]
        for label in (0,1):
            selected=[r for r in records if r['record']['label']==label]
            assert len(selected)==(720 if label else 6480)
            # Retain complete original source row fields; group by stored voice identity.
            groups={'all':selected}
            for variant in range(3):groups['variant:'+str(variant)]=[r for r in selected if r['record']['variant']==variant]
            for engine in sorted({r['record']['engine'] for r in selected}):
                groups['engine:'+engine]=[r for r in selected if r['record']['engine']==engine]
            for key,group in groups.items():
                assert group
                summary.append(dict(label=label,group=key,rows=len(group),maximum_five_peak=max(r['five_mean_peak'] for r in group),median_five_peak=float(np.median([r['five_mean_peak'] for r in group]))))
        result=dict(input_quantization=[float(detector.input['quantization'][0]),int(detector.input['quantization'][1])],output_quantization=[float(detector.output['quantization'][0]),int(detector.output['quantization'][1])],target=target,model_sha256=p['models'][target]['sha256'],rows=7200,records_sha256=sha(dest/'records.jsonl'),raw_archive_sha256=sha(dest/'raw-uint8.bin'),summary=summary)
        (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n');summaries[target]=result
    report=dict(schema='hey-training-stream-diagnostic-completion-v1',plan_sha256=sha(PLAN),summaries=summaries,training_updated=False,threshold_selected=False,acceptance_accessed=False,limits='Complete training support only, no independent recognition/FAR result or isolated causal claim.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));a=parser.parse_args();{'register':register,'run':run}[a.action]()
