"""Interleaved frozen C161/BN on the same 144 natural-language clips isolated and continuous."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from collect_multispeaker_formal_v1 import prepare
import run_c161_binary40_parallel_v1 as parallel
from run_parallel_context_streams_v1 import material
from uint8_event_policy_v1 import event_steps

ROOT = Path('research/wakeword_p4')
RESULTS = ROOT/'results'
SOURCE = RESULTS/'4090-parallel-context-streams-plan-v1.json'
REFERENCE = RESULTS/'4090-c161-channel-bn-events-streams-v1.jsonl.gz'
STEM = '4090-c161-bn-natural-language'
PLAN = RESULTS/(STEM+'-plan-v1.json')
sys.path.insert(0, str((ROOT/'training').resolve()))
from features import features


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(p, d):
    with p.open('x') as f:
        json.dump(d, f, indent=2)
        f.write('\n')


def plan():
    source = json.loads(SOURCE.read_bytes())
    export = RESULTS/'4090-broadcast-channel-bn-joint-export-original-v1.json'
    candidate = json.loads(export.read_bytes())['model']
    assert candidate['sha256'] == 'dc4e5874141828ae6bf88d03011411dbb4a617c262215467fae09fec3526945c'
    files = dict(source['files'])
    for f in [SOURCE, REFERENCE, export, Path(__file__), Path(parallel.__file__), ROOT/'training/features.py',
              Path(__file__).with_name('run_parallel_context_streams_v1.py'),
              Path(__file__).with_name('collect_multispeaker_formal_v1.py'),
              Path(__file__).with_name('uint8_event_policy_v1.py')]:
        files[str(f)] = sha(f)
    files[candidate['path']] = candidate['sha256']
    files[parallel.MODELS['c161']] = 'c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
    for name, digest in files.items():
        assert sha(name) == digest
    write(PLAN, dict(files=files, candidate=candidate, configurations=['isolated','continuous'], gap_samples=8000,
                     prime_samples=81600, policy='All original 3x48 consumed clips and original physical streams. '
                     'Run actual C161 and channel BN interpreters interleaved, native 480 sample returns. Original '
                     'two-parent arbitration and fixed operating points. Compare clip identity, complete returned '
                     'events and everyday/common-name collisions across history. Full priming and gap events kept. '
                     'No new data, exposure, threshold, optimizer or P4 measurement.'))


def infer(pcm):
    full = np.r_[np.zeros(81600,dtype=np.int16),pcm]
    raw = features(full)
    values,support,_ = prepare(raw,len(full))
    clock = np.minimum(((support+479)//480)*480,len(full))-81600
    scores,events,emitted,suppressed,timing = parallel.dual(values,clock)
    for k in scores:
        assert events[k] == [dict(step=s,relative_sample=int(clock[s-1])) for s in event_steps(scores[k],235/256)]
    def renamed(v):
        return [dict(e,model=('channel_bn' if e['model']=='binary40' else e['model'])) for e in v]
    return dict(c161=events['c161'],channel_bn=events['binary40'],pair=renamed(emitted)),renamed(suppressed),timing,{
        'c161':scores['c161'],'channel_bn':scores['binary40'],'clock':clock}


def run():
    cfg=json.loads(PLAN.read_bytes())
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)]) == PLAN.read_bytes()
    for path,digest in cfg['files'].items():
        assert sha(path)==digest,path
    # This process owns both interpreters. Historical helper's second slot has a legacy key.
    parallel.MODELS['binary40'] = cfg['candidate']['path']
    with gzip.open(REFERENCE,'rt') as f:
        refs={row['name']:row for row in map(json.loads,f) if row['cohort']=='context'}
    rows,streams,arrays=[],[],{}
    for stream in json.loads(SOURCE.read_bytes())['streams']:
        name=stream['name']
        pcm,timeline=material(stream,8000)
        events,suppressed,timing,trace=infer(pcm)
        assert events==refs[name]['events'] and suppressed==refs[name]['suppressed']
        for k,v in trace.items():arrays[f'{name}_stream_{k}']=v
        streams.append(dict(name=name,events=events,suppressed=suppressed,timing=timing))
        clips={v['text_index']:v for v in timeline if v['kind']=='clip'}
        for item in stream['items']:
            i=item['text_index']
            audio,sr=sf.read(item['wav'],dtype='int16')
            assert sr==16000 and audio.ndim==1 and hashlib.sha256(audio.astype('<i2').tobytes()).hexdigest()==item['pcm_sha256']
            isolated,ignored,measurement,values=infer(audio)
            assert isolated['c161']==item['isolated_events']['c161']
            for k,v in values.items():arrays[f'{name}_{i}_{k}']=v
            clip=clips[i]
            inside={k:[e for e in ev if clip['start']<e['relative_sample']<=clip['end']] for k,ev in events.items()}
            rows.append(dict(stream=name,index=i,text=item['request']['text'],group=item['request']['group'],
                             pcm_sha256=item['pcm_sha256'],isolated=isolated,isolated_suppressed=ignored,
                             continuous=inside,continuous_clip_start=clip['start'],continuous_clip_end=clip['end'],
                             isolated_timing=measurement))
    assert len(rows)==144 and len(streams)==3
    archive=RESULTS/(STEM+'-scores-v1.npz')
    with archive.open('xb') as f:np.savez_compressed(f,**arrays)
    with np.load(archive,allow_pickle=False) as z:
        assert set(z.files)==set(arrays) and all(np.array_equal(v,z[k]) for k,v in arrays.items())
    write(RESULTS/(STEM+'-original-v1.json'),dict(plan_sha256=sha(PLAN),scores_sha256=sha(archive),
          streams=streams,records=rows,policy=cfg['policy']))


if __name__=='__main__':
    {'plan':plan,'run':run}[sys.argv[1]]()
