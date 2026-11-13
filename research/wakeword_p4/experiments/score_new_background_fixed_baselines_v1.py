"""Full original42-recording fixed HiESP/v30 baselines; no tuning or candidate."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.stats import chi2

ROOT=Path('research/wakeword_p4')
sys.path.insert(0,str(ROOT/'training'))
from evaluate import Detector,event_indices,smooth
from features import features

OUT=Path('/tmp/nyx-newbg-base-v1')
FREEZE=ROOT/'results/4090-new-background-fixed-baseline-execution-pins-v1.json'
MANIFEST=Path('.local/wakeword/prospective-background-whole-waveform-v1/fixed-baseline-replay-manifest-v1.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_freeze():
    parent=ROOT/'results/4090-joe-fixed-baseline-execution-pins-v1.json'
    pins=json.loads(parent.read_text())
    for item in list(pins['files'].values())+list(pins['gate_files'].values()):assert sha(item['path'])==item['sha256']
    m=json.loads(MANIFEST.read_text())
    assert len(m['records'])==42 and m['source_samples']==455204864
    assert m['continuous_recordings'] and not m['inserted_intermediate_gaps']
    for r in m['records']:
        assert sha(MANIFEST.parent/r['path'])==r['sha256']
    pins.update(schema='new-background-fixed-baseline-execution-pins-v1',parent_pins_sha256=sha(parent),
                executor_sha256=sha(__file__),manifest=dict(path=str(MANIFEST),sha256=sha(MANIFEST)),
                intended_records=42,original_samples=455204864,original_hours=455204864/16000/3600,
                execution_authorized='Full42 new DEVELOPMENT backgrounds fixed official/v30 only',
                procedure='Official4workers with600s per-recording engine deadline, no retries; then continuous original frontend/INT8 v30 twice per full recording/reset only at recording start; exact repeatability. Every event counted, original samples only exposure. No slicing/annotation filtering, inserted gaps or duration extension. Incomplete run has no aggregate quality summary.',
                candidate_execution_authorized=False,acceptance_execution_authorized=False)
    with FREEZE.open('x') as f:json.dump(pins,f,indent=2);f.write('\n')
    print(json.dumps(dict(freeze_sha256=sha(FREEZE),records=42,new_scores=False)))


def score():
    pins=json.loads(FREEZE.read_text());assert sha(__file__)==pins['executor_sha256']
    for item in list(pins['files'].values())+list(pins['gate_files'].values())+[pins['manifest']]:assert sha(item['path'])==item['sha256']
    files=pins['files'];manifest=Path(pins['manifest']['path']);m=json.loads(manifest.read_text());rows=m['records']
    assert len(rows)==42 and sum(r['samples'] for r in rows)==455204864
    OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_bytes(FREEZE.read_bytes())
    def replay(item):
        i,row=item
        args=[sys.executable,files['runner']['path'],'--manifest',str(manifest),'--record',row['path'],
              '--threshold',str(pins['official_threshold']),'--timeout-seconds','600','--output',str(OUT/str(i))]
        for key in ['qemu','elf','nm','flash']:args+=['--'+key,files[key]['path']]
        with (OUT/f'{i}-runner.log').open('wb') as log:proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        return i,None if proc.returncode else json.loads((OUT/str(i)/'report.json').read_text())
    reports={};failed=[]
    with ThreadPoolExecutor(4) as pool:
        for future in as_completed([pool.submit(replay,item) for item in enumerate(rows)]):
            i,report=future.result()
            if report is None:failed.append(i)
            else:reports[i]=report
            print(json.dumps(dict(official_completed=len(reports),failures=len(failed),planned=42)),flush=True)
    if failed:
        (OUT/'failures.json').write_text(json.dumps(failed)+'\n')
        raise RuntimeError('Incomplete background official replay; no retries or aggregate summary')
    detector=Detector(files['v30']['path']);results=[]
    assert detector.input['dtype']==np.int8 and detector.output['dtype']==np.uint8
    assert tuple(detector.input['shape'])==(1,3,40) and tuple(detector.output['shape'])==(1,1)
    for i,row in enumerate(rows):
        official=reports[i]
        assert official['record']==row and official['manifest_sha256']==pins['manifest']['sha256']
        assert official['firmware_stage']==104 and official['firmware_crc32_matched']
        assert official['threshold_getter_after_set']==pins['official_getter']
        assert official['input_samples']==row['samples'] and official['pcm_sha256']==row['pcm_sha256']
        assert official['calls']==(row['samples']+511)//512 and official['final_chunk_zero_pad_samples']==(-row['samples'])%512
        assert official['source_sha256']==files['runner']['sha256']
        for key in ['qemu','elf','flash']:assert official[key+'_sha256']==files[key]['sha256']
        path=manifest.parent/row['path'];assert sha(path)==row['sha256']
        pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
        raw=features(pcm);detector.reset();scores=smooth(detector.predict(raw),5)
        detector.reset();np.testing.assert_array_equal(scores,smooth(detector.predict(raw),5))
        assert len(scores)==((len(pcm)+159)//160-2)//3-4 and np.isfinite(scores).all() and np.all((0<=scores)&(scores<=1))
        trace=OUT/f'{i}-v30-scores.npy';np.save(trace,scores)
        times=[float((3*(j+4)+5)*.01) for j in event_indices(scores,pins['v30_threshold'])]
        results.append(dict(record=row,v30=dict(event_seconds=times,score_sha256=sha(trace),reset_repeatable=True),
                            hiesp=dict(event_seconds=[e['audio_seconds'] for e in official['events']],report_sha256=sha(OUT/str(i)/'report.json'))))
        print(json.dumps(dict(v30_completed=i+1,planned=42)),flush=True)
    hours=sum(r['samples'] for r in rows)/16000/3600
    summary={}
    for engine in ['v30','hiesp']:
        count=sum(len(r[engine]['event_seconds']) for r in results)
        summary[engine]=dict(events=count,original_hours=hours,events_per_hour=count/hours,
                             upper_one_sided95_per_hour=float(.5*chi2.ppf(.95,2*(count+1))/hours))
    report=dict(role='Full new DEVELOPMENT background fixed baselines',execution_pins_sha256=sha(FREEZE),
                summaries=summary,records=results,original_samples=455204864,
                candidate_scored=False,acceptance_scored=False,thresholds_selected=False,
                limits='Unscreened full recordings, every event conservatively counted. Nominal Poisson bound does not remove clustering. Fixed baseline operating points are not matched population FAR. Original annotation failures retained. No candidate promotion or final acceptance.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--write-freeze']:write_freeze()
    elif sys.argv[1:]==['--score']:score()
    else:raise SystemExit('Use --write-freeze or --score')
