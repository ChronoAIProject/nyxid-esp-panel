"""Pin then collect full development raw streams for paired candidates and v30."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
from uint8_stream_detector_v1 import Uint8StreamDetector
from generate_hey_development_native_v1 import sha
from audit_hey_training_pcm_history_v1 import pcm

RESULTS=Path('research/wakeword_p4/results')
BASE=Path('.local/wakeword')
SYNTHETIC=BASE/'hey-development-replay-v1.json'
BACKGROUND=BASE/'hey-paired-earnings-development-v1/manifest.json'
FREEZE=RESULTS/'4090-hey-development-custom-execution-v1.json'
OUT=BASE/'hey-development-custom-streams-v1'


def versions():return {p:importlib.metadata.version(p) for p in ('numpy','scipy','soundfile','ai-edge-litert','pymicro-features')}


def register():
    candidates=RESULTS/'4090-hey-v2-fixed-final-candidates-v1.json'
    models=json.loads(candidates.read_text())['models']
    models['v30']=dict(path=str(BASE/'model-gpu-v30-ami-extra/hey_nyx.tflite'),bytes=54440,sha256='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c')
    for m in models.values():assert sha(m['path'])==m['sha256'] and Path(m['path']).stat().st_size==54440
    hist=RESULTS/'4090-hey-earnings-history-overlap-v1.json';h=json.loads(hist.read_text())
    assert not h['matches'] and h['no_exact_encoded_or_pcm_hash_matches'] and h['manifest_sha256']==sha(BACKGROUND)
    assert sha(h['inventory_path'])==h['inventory_sha256']
    synthetic=json.loads(SYNTHETIC.read_text());background=json.loads(BACKGROUND.read_text())
    assert len(synthetic['records'])==360 and len(background['records'])==12
    assert background['all_complete'] and background['integrity_preparation_passed'] and background['hours']>=8
    prep=RESULTS/'4090-hey-earnings-development-complete-preparation-v1.json';p=json.loads(prep.read_text())
    assert p['manifest_sha256']==sha(BACKGROUND) and p['transcript_alias_matches']==0
    dh=RESULTS/'4090-hey-development-pcm-history-v1.json';d=json.loads(dh.read_text())
    assert d['no_unexpected_exact_hash_matches'] and not d['matches']
    assert sha(d['inventory_path'])==d['inventory_sha256']
    paths=[Path(__file__),candidates,SYNTHETIC,BACKGROUND,hist,prep,dh,Path(h['inventory_path']),Path(d['inventory_path'])]
    paths += [Path(__file__).with_name(n) for n in ('uint8_stream_detector_v1.py','uint8_event_policy_v1.py','hey_development_metrics_v1.py','hey_development_scoring_contract_v1.md','calibrate_hey_development_custom_v1.py','generate_hey_development_native_v1.py','audit_hey_training_pcm_history_v1.py')]
    paths += [Path(__file__).resolve().parents[1]/'training/features.py']
    # Bind installed runtime/frontend source and native libraries, not versions alone.
    for package in ('ai_edge_litert','pymicro_features'):
        spec=importlib.util.find_spec(package);assert spec and spec.origin
        folder=Path(spec.origin).parent
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.so'))
    for row in synthetic['records']:
        assert sha(row['path'])==row['sha256'];paths.append(Path(row['path']))
    for row in background['records']:
        assert row['role']=='development' and sha(row['wav_path'])==row['wav_sha256']
        assert pcm(row['wav_path'],row['samples'])==row['pcm_sha256']
        assert sha(row['frontend_path'])==row['frontend_sha256']
        paths.extend([Path(row['wav_path']),Path(row['frontend_path'])])
    r=dict(schema='hey-development-custom-execution-v1',models=models,versions=versions(),files={str(p):sha(p) for p in paths},synthetic_manifest=str(SYNTHETIC),background_manifest=str(BACKGROUND),authorizes='One full DEVELOPMENT raw stream collection for each frozen model, complete-input reset repeat, no acceptance or promotion.',event_policy='uint8-sum-binary64-v1',background_overlap_admitted=True,limits='Bounded exact history and declared TTS/source reservations, not physical/pretraining/subclip independence. No filtering or retries. Separate registered calibration after collection.')
    with FREEZE.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(execution_sha256=sha(FREEZE),scored=False)))


def score(model_id):
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(FREEZE)])
    assert remote==FREEZE.read_bytes();f=json.loads(remote)
    assert versions()==f['versions']
    for p,h in f['files'].items():assert sha(p)==h
    model=f['models'][model_id];assert sha(model['path'])==model['sha256']
    dest=OUT/model_id;dest.mkdir(parents=True,exist_ok=False)
    detector=Uint8StreamDetector(model['path']);phases={}
    for phase,manifest in (('synthetic',SYNTHETIC),('background',BACKGROUND)):
        rows=json.loads(manifest.read_text())['records'];group=dest/phase;group.mkdir();results=[]
        with (group/'records.jsonl').open('x') as log:
            for i,row in enumerate(rows):
                path=Path(row['path'] if phase=='synthetic' else row['wav_path'])
                assert pcm(path,row['samples'])==row['pcm_sha256']
                if phase=='synthetic':
                    values,rate=sf.read(path,dtype='int16');assert rate==16000
                    raw=features(values)
                else:
                    raw=np.load(row['frontend_path'],allow_pickle=False)
                    assert raw.shape==(row['frames'],40)
                assert raw.shape==(((row['samples']+159)//160-2),40)
                detector.reset();first=detector.predict(raw);detector.reset();second=detector.predict(raw)
                np.testing.assert_array_equal(first,second)
                assert first.dtype==np.uint8 and first.shape==(len(raw)//3,)
                trace=group/f'{i}-raw.npy';np.save(trace,first,allow_pickle=False)
                record=dict(record=row,trace_path=str(trace),trace_sha256=sha(trace),raw_steps=len(first),reset_repeat_exact=True)
                results.append(record);log.write(json.dumps(record)+'\n');log.flush()
                print(json.dumps(dict(model=model_id,phase=phase,completed=i+1)),flush=True)
        phases[phase]=dict(manifest_sha256=sha(manifest),records=results)
    report=dict(schema='hey-development-custom-stream-report-v1',model_id=model_id,model_sha256=model['sha256'],execution_sha256=sha(FREEZE),phases=phases,threshold_selected=False,acceptance_scored=False,candidate_promoted=False)
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','score'));p.add_argument('--model',choices=('hey_willow','hey_nyx','v30'));a=p.parse_args()
    if a.action=='register':register()
    else:
        if not a.model:p.error('--model required')
        score(a.model)
