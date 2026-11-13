"""Frozen official HiESP/v30 Joe development baselines only; no candidate tuning."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT=Path('research/wakeword_p4')
sys.path.insert(0,str(ROOT/'training'))
from evaluate import Detector, event_indices, smooth
from features import features

FREEZE=ROOT/'results/4090-joe-fixed-baseline-execution-pins-v1.json'
OUT=Path('/tmp/nyx-joe-base-v1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p:importlib.metadata.version(p) for p in ['numpy','soundfile','ai-edge-litert','pymicro-features']}


def write_freeze():
    inventory=ROOT/'results/4090-joe-official-runtime-inventory-v1.json'
    inv=json.loads(inventory.read_text())
    for entry in inv['files'].values():assert sha(entry['path'])==entry['sha256']
    gate_files={
        'main_source_audit':ROOT/'results/4090-joe-known-source-overlap-v1.json',
        'source_supplement':ROOT/'results/4090-joe-known-source-overlap-supplement-v1.json',
        'native_summary':ROOT/'results/4090-joe-development-native-transport-summary-v1.json',
        'render_summary':ROOT/'results/4090-joe-development-render-summary-v1.json',
        'selection_protocol':ROOT/'experiments/joe_mike_selection_protocol_v1.md',
        'runtime_inventory':inventory,
    }
    assert not json.loads(gate_files['main_source_audit'].read_text())['known_source_overlap_found']
    assert json.loads(gate_files['source_supplement'].read_text())['no_known_exact_hash_conflicts']
    assert json.loads(gate_files['native_summary'].read_text())['mechanical_pack_qualified']
    assert json.loads(gate_files['render_summary'].read_text())['mechanically_qualified']
    plan=dict(schema='joe-fixed-baselines-execution-pins-v1',files=inv['files'],
              gate_files={k:dict(path=str(p),sha256=sha(p)) for k,p in gate_files.items()},
              executor_sha256=sha(__file__),versions=versions(),workers=4,
              official_threshold=.63,official_getter=.62496001,v30_threshold=.91796875,smoothing=5,
              intended_records=360,execution_authorized='Joe development fixed official/v30 baselines only',
              candidate_execution_authorized=False,acceptance_execution_authorized=False,
              procedure='All360 official replays, no retries. Require complete byte/CRC/call/stage evidence. Then original v30 streaming INT8 traces twice per clip with fresh reset and exact repeatability. Save every event and signed generated/RIR latency; no thresholds selected. Any missing/error prevents combined summary. All failure artifacts retained.',
              source_scope='Known exact original-container/PCM/hash conflicts excluded by recorded audits. Sharedrdh parent and unknown transformed/partial/person/pretraining overlap remain limits, not certified independence.')
    with FREEZE.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps(dict(freeze_sha256=sha(FREEZE),records=360,new_scores=False)))


def summarize(rows,engine):
    target='hey_nyx' if engine=='v30' else 'hi_esp'
    positives=[r for r in rows if r['record']['target']==target]
    def rate(group):return sum(bool(r[engine]['post_source_events']) for r in group)/len(group)
    result=dict(positive_clips=len(positives),post_source_activated=sum(bool(r[engine]['post_source_events']) for r in positives),
        activation_rate=rate(positives),macro=float(np.mean([rate([r for r in positives if r['record']['variant']==v]) for v in range(3)])),
        pre_source_positive_events=sum(sum(t<1.8 for t in r[engine]['event_seconds']) for r in positives),
        standalone_rate=rate([r for r in positives if ', ' not in r['record']['text']]),
        command_rate=rate([r for r in positives if ', ' in r['record']['text']]))
    for name,group in [('common',[r for r in rows if r['record']['target']=='confusion']),
                       ('cross_target',[r for r in rows if r['record']['target'] not in ('confusion',target)])]:
        result[name]=dict(clips=len(group),activated_clips=sum(bool(r[engine]['event_seconds']) for r in group),events=sum(len(r[engine]['event_seconds']) for r in group))
    result['conditions']={str(v):dict(clips=len([r for r in positives if r['record']['variant']==v]),activation_rate=rate([r for r in positives if r['record']['variant']==v])) for v in range(3)}
    return result


def score():
    freeze=json.loads(FREEZE.read_text());assert sha(__file__)==freeze['executor_sha256'] and versions()==freeze['versions']
    for entry in list(freeze['files'].values())+list(freeze['gate_files'].values()):assert sha(entry['path'])==entry['sha256']
    files=freeze['files'];manifest_path=Path(files['development_manifest']['path']);manifest=json.loads(manifest_path.read_text());rows=manifest['records']
    assert len(rows)==360 and all(r['voice']=='joe' and r['role']=='development-paired-validation' for r in rows)
    for row in rows:
        path=manifest_path.parent/row['path'];assert sha(path)==row['sha256']
    OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_bytes(FREEZE.read_bytes())
    def replay(item):
        i,row=item
        args=[sys.executable,files['runner']['path'],'--manifest',str(manifest_path),'--record',row['path'],
              '--threshold',str(freeze['official_threshold']),'--output',str(OUT/str(i))]
        for name in ['qemu','elf','nm','flash']:args+=['--'+name,files[name]['path']]
        with (OUT/f'{i}-runner.log').open('wb') as log:
            proc=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:return i,None
        return i,json.loads((OUT/str(i)/'report.json').read_text())
    reports={};failures=[]
    with ThreadPoolExecutor(freeze['workers']) as pool:
        for future in as_completed([pool.submit(replay,item) for item in enumerate(rows)]):
            i,report=future.result()
            if report is None:failures.append(i)
            else:reports[i]=report
            print(json.dumps(dict(official_completed=len(reports),failures=len(failures),planned=360)),flush=True)
    if failures:
        (OUT/'failures.json').write_text(json.dumps(failures)+'\n')
        raise RuntimeError('Incomplete official replay, preserve all artifacts; no combined summary/retries')
    detector=Detector(files['v30']['path'])
    assert detector.input['dtype']==np.int8 and detector.output['dtype']==np.uint8
    assert tuple(detector.input['shape'])==(1,3,40) and tuple(detector.output['shape'])==(1,1)
    results=[]
    for i,row in enumerate(rows):
        official=reports[i]
        assert official['record']==row and official['manifest_sha256']==files['development_manifest']['sha256']
        assert official['source_sha256']==files['runner']['sha256']
        assert official['firmware_stage']==104 and official['firmware_crc32_matched']
        assert official['input_samples']==row['samples'] and official['pcm_sha256']==row['pcm_sha256']
        assert official['calls']==(row['samples']+511)//512 and official['final_chunk_zero_pad_samples']==(-row['samples'])%512
        assert official['threshold_getter_after_set']==freeze['official_getter']
        for name in ['qemu','elf','flash']:assert official[name+'_sha256']==files[name]['sha256']
        pcm,rate=sf.read(manifest_path.parent/row['path'],dtype='int16');assert rate==16000
        raw=features(pcm);detector.reset();scores=smooth(detector.predict(raw),5)
        detector.reset();np.testing.assert_array_equal(scores,smooth(detector.predict(raw),5))
        assert len(scores)==((len(pcm)+159)//160-2)//3-4 and np.isfinite(scores).all() and np.all((0<=scores)&(scores<=1))
        trace=OUT/f'{i}-v30-scores.npy';np.save(trace,scores)
        nyx_times=[float((3*(j+4)+5)*.01) for j in event_indices(scores,freeze['v30_threshold'])]
        esp_times=[e['audio_seconds'] for e in official['events']]
        def events(times):
            return dict(event_seconds=times,post_source_events=[t for t in times if t>=1.8],
                all_event_signed_generated_end_ms=[(t-row['generated_output_end_sample']/16000)*1000 for t in times],
                all_event_signed_full_rir_end_ms=[(t-row['full_rir_end_sample']/16000)*1000 for t in times])
        results.append(dict(record=row,v30=dict(events(nyx_times),score_sha256=sha(trace),reset_repeatable=True),
                            hiesp=dict(events(esp_times),report_sha256=sha(OUT/str(i)/'report.json'))))
        if i%30==0:print(json.dumps(dict(v30_completed=i+1,planned=360)),flush=True)
    report=dict(role='New Joe development fixed-baseline synthetic reference only',execution_pins_sha256=sha(FREEZE),
                summaries={engine:summarize(results,engine) for engine in ['v30','hiesp']},records=results,
                candidate_scored=False,acceptance_scored=False,thresholds_selected=False,background_scored=False,
                limits='One development voice; full-generated/container/RIR end is a synthetic reference, not wake-prefix/phonetic truth. Fixed operating points are not matched population FAR. Shared Joe/Mike parent, unknown physical/pretraining overlap. No candidate or quality promotion.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summaries']),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--write-freeze']:write_freeze()
    elif sys.argv[1:]==['--score']:score()
    else:raise SystemExit('Use --write-freeze or --score')
