"""Pin then score the sole completed recovered export on reserved DEVELOPMENT PCM."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT=Path('research/wakeword_p4')
sys.path.insert(0,str(ROOT/'training'))
from evaluate import Detector,smooth
from features import features

BASE=Path('.local/wakeword')
MODEL=BASE/'model-complete-recovery-recovered-formal-v1'
OUT=BASE/'recovered-new-development-v1'
FREEZE=ROOT/'results/4090-recovered-new-development-execution-pins-v1.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def versions():
    return {p:importlib.metadata.version(p) for p in ['numpy','scipy','soundfile','ai-edge-litert','pymicro-features']}


def prerequisites():
    required=[MODEL/'training.json',BASE/'complete-recovery-recovered-formal-v1-export-audit.json',
              BASE/'complete-recovery-control-formal-v1-reused-development-exploratory-calibration-v1.json',
              BASE/'complete-recovery-recovered-formal-v1-reused-development-exploratory-calibration-v1.json',
              Path('/tmp/nyx-joe-base-v1/report.json'),Path('/tmp/nyx-newbg-base-v1/report.json')]
    return required,[str(p) for p in required if not p.is_file()]


def write_freeze():
    required,missing=prerequisites()
    if missing:raise RuntimeError('Original training/postprocessing or baselines incomplete: '+', '.join(missing))
    training=json.loads((MODEL/'training.json').read_text())
    assert training['steps']==120000 and training['recovered_erased_complete_coverage'] is True
    assert training['model_sha256']==sha(MODEL/'hey_nyx.tflite')
    assert training['batchnorm_initial']==training['batchnorm_selected']
    audit=json.loads(required[1].read_text())
    assert audit['model_sha256']==training['model_sha256'] and audit['training_sha256']==sha(MODEL/'training.json')
    assert audit['all_saved_weights_finite'] and audit['frozen_bn_unchanged']
    # Authenticate original experiment pins; do not alter a live training source.
    original_pins=ROOT/'results/4090-complete-recovery-execution-pins-v1.json'
    for p,h in json.loads(original_pins.read_text())['file_sha256'].items():assert sha(p)==h
    inventory_path=ROOT/'results/4090-joe-official-runtime-inventory-v1.json'
    inv=json.loads(inventory_path.read_text())
    for entry in inv['files'].values():assert sha(entry['path'])==entry['sha256']
    sources={
        'synthetic':BASE/'joe-development-render-v1/replay-manifest.json',
        'background':BASE/'prospective-background-whole-waveform-v1/fixed-baseline-replay-manifest-v1.json'}
    gates=[ROOT/'results/4090-joe-known-source-overlap-v1.json',
           ROOT/'results/4090-joe-known-source-overlap-supplement-v1.json',
           ROOT/'experiments/joe_mike_selection_protocol_v1.md',inventory_path,original_pins]
    assert not json.loads(gates[0].read_text())['known_source_overlap_found']
    assert json.loads(gates[1].read_text())['no_known_exact_hash_conflicts']
    synthetic_baseline=json.loads(required[4].read_text());background_baseline=json.loads(required[5].read_text())
    assert len(synthetic_baseline['records'])==360 and len(background_baseline['records'])==42
    assert not synthetic_baseline['candidate_scored'] and not background_baseline['candidate_scored']
    for baseline,name in [(synthetic_baseline,'joe'),(background_baseline,'new-background')]:
        p=ROOT/f'results/4090-{name}-fixed-baseline-execution-pins-v1.json'
        assert baseline['execution_pins_sha256']==sha(p)
        old=json.loads(p.read_text())
        for entry in list(old['files'].values())+list(old['gate_files'].values()):
            assert sha(entry['path'])==entry['sha256']
        gates.append(p)
    assert [r['record'] for r in synthetic_baseline['records']]==json.loads(sources['synthetic'].read_text())['records']
    assert [r['record'] for r in background_baseline['records']]==json.loads(sources['background'].read_text())['records']
    files={str(p):dict(size=p.stat().st_size,sha256=sha(p)) for p in required+gates+list(sources.values())+[MODEL/'hey_nyx.tflite',MODEL/'best.weights.h5']}
    for key in ['detector','frontend']:
        p=Path(inv['files'][key]['path']);files[str(p)]=dict(size=p.stat().st_size,sha256=sha(p))
    freeze=dict(schema='sole-recovered-new-development-execution-pins-v1',files=files,
                manifests={k:str(v) for k,v in sources.items()},model_sha256=training['model_sha256'],
                executor_sha256=sha(__file__),
                calibrator_sha256=sha(ROOT/'experiments/calibrate_recovered_new_development_v1.py'),
                versions=versions(),expected_records={'synthetic':360,'background':42},
                authorizes='Sole original completed recovered export DEVELOPMENT traces only; no threshold/acceptance/promotion',
                procedure='Original continuous full-waveform frontend/streaming INT8 with smoothing5. Reset at clip/recording start only, repeat each entire prediction exactly with fresh reset. Preserve every error/partial output, no retries/alternate checkpoint. Reuse completed exact same-PCM official/v30 baseline reports; no official rerun or new thresholds here.')
    with FREEZE.open('x') as f:json.dump(freeze,f,indent=2);f.write('\n')
    print(json.dumps(dict(freeze_sha256=sha(FREEZE),model_sha256=training['model_sha256'],candidate_scored=False)))


def score():
    freeze=json.loads(FREEZE.read_text())
    assert sha(__file__)==freeze['executor_sha256'] and versions()==freeze['versions']
    for p,e in freeze['files'].items():assert sha(p)==e['sha256'] and Path(p).stat().st_size==e['size']
    detector=Detector(MODEL/'hey_nyx.tflite')
    assert detector.input['dtype']==np.int8 and detector.output['dtype']==np.uint8
    assert tuple(detector.input['shape'])==(1,3,40) and tuple(detector.output['shape'])==(1,1)
    OUT.mkdir(exist_ok=False);(OUT/'freeze.json').write_bytes(FREEZE.read_bytes())
    phases={}
    for phase,path in freeze['manifests'].items():
        manifest=Path(path);m=json.loads(manifest.read_text());rows=m['records']
        assert len(rows)==freeze['expected_records'][phase]
        assert all(r['role'].startswith('development-') and r['split']=='validation' for r in rows)
        group=OUT/phase;group.mkdir();completed=[]
        with (group/'records.jsonl').open('x') as log:
            for i,row in enumerate(rows):
                path=manifest.parent/row['path'];assert sha(path)==row['sha256']
                pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
                assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
                raw=features(pcm);detector.reset();scores=smooth(detector.predict(raw),5)
                detector.reset();np.testing.assert_array_equal(scores,smooth(detector.predict(raw),5))
                assert len(scores)==((len(pcm)+159)//160-2)//3-4 and np.isfinite(scores).all() and np.all((0<=scores)&(scores<=1))
                trace=group/f'{i}-scores.npy';np.save(trace,scores)
                record=dict(record=row,score_sha256=sha(trace),scores=len(scores),reset_repeatable=True)
                completed.append(record);log.write(json.dumps(record)+'\n');log.flush()
                print(json.dumps(dict(phase=phase,completed=i+1,planned=len(rows))),flush=True)
        phases[phase]=dict(manifest_sha256=sha(manifest),records=completed)
    report=dict(model_sha256=freeze['model_sha256'],execution_pins_sha256=sha(FREEZE),phases=phases,
                smoothing=5,threshold_selected=False,acceptance_scored=False,candidate_promoted=False,
                limits='Complete new DEVELOPMENT traces only; separate preregistered calibration required. Synthetic generation/RIR boundaries are not phonetic truth; shared parent/unknown partial/person/pretraining overlap limits persist.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    if sys.argv[1:]==['--check-prerequisites']:
        _,missing=prerequisites();print(json.dumps(dict(ready=not missing,missing=missing,candidate_scored=False)))
    elif sys.argv[1:]==['--write-freeze']:write_freeze()
    elif sys.argv[1:]==['--score']:score()
    else:raise SystemExit('Use --check-prerequisites, --write-freeze or --score')
