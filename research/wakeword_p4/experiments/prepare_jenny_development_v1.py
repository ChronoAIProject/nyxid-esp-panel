"""Freeze prospective development material and duration observer; no inference."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import onnx
from piper_phonemize import phonemize_espeak

RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-jenny-development-material-plan-v1.json'
OUT=Path('.local/wakeword/jenny-development-observer-v1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    identity=RESULTS/'4090-author-jenny-model-identity-v1.json';r=json.loads(identity.read_text())
    model=Path(r['files']['jenny.onnx']['local_path']);config=Path(r['files']['jenny.onnx.json']['local_path'])
    assert sha(model)==r['files']['jenny.onnx']['sha256'] and sha(config)==r['files']['jenny.onnx.json']['sha256']
    cfg=json.loads(config.read_text());ids=cfg['phoneme_id_map'];texts={};requests=[]
    phrases=['Hey Willow','Hey Nicks','Hi E S P','Hey','Hello','Hey William','Hey pillow','Hey Nick','Hey Next','Hey Mix','Hey Muse']
    carriers=dict(bare='.',light=', turn on the light.',time=', what time is it?',timer=', set a timer.')
    for phrase in phrases:
        for carrier,suffix in carriers.items():
            text=phrase+suffix;sentences=phonemize_espeak(text,cfg['espeak']['voice'])
            assert len(sentences)==1
            phones=sentences[0];assert all(p in ids for p in phones)
            tokens=list(ids['^'])+list(ids['_'])
            cut=phones.index('.' if carrier=='bare' else ',')
            prefix=len(tokens)+sum(len(ids[p])+len(ids['_']) for p in phones[:cut])
            for phone in phones:tokens.extend(ids[phone]);tokens.extend(ids['_'])
            tokens.extend(ids['$'])
            texts[text]=dict(phonemes=phones,token_ids=tokens,prefix_token_count=prefix)
            for scale in (.9,1.,1.1):
                requests.append(dict(order=len(requests),phrase=phrase,carrier=carrier,text=text,length_scale=scale,
                    noise_scale=.667,noise_scale_w=.8,role='development-material-preparation'))
    files={str(p):sha(p) for p in [Path(__file__),identity,model,config,RESULTS/'4090-author-jenny-static-duration-feasibility-v1.json',RESULTS/'4090-hey-broad-fresh-development-source-review-v1.json']}
    result=dict(schema='jenny-development-material-plan-v1',files=files,requests=requests,texts=texts,
        versions={n:importlib.metadata.version(n) for n in ['onnx','piper-phonemize','numpy','onnxruntime']},
        voice='Jenny (Dioco), author-direct checkpoint; single declared voice, not132independent voices',
        role='Prospective development only; excluded from training and independent acceptance',
        lineage='Author claims scratch Jenny corpus; distinct export from Piper Jenny. Physical-speaker/pretraining independence not independently certified. Mac341ffbf bounded metadata found no prior use; not universal nonuse proof.',
        attribution='Jenny (Dioco); retain original dataset/license receipts',
        observer='Add /Ceil_output_0 as second graph output only; original nodes, weights, inputs and waveform output retained. Never overwrite original model.',
        quality='Save full native float, resampled float,PCM, tokens and actual integer durations. Require finite/mono/nonempty/.2..10s/RMS>=.001/no fullscale or PCM endpoints,full ceil resample length; actual durations length=tokens,sum*256=fullnative waveform. No crop/normalization/retry/ASR selection. Any failure retained and reported.',
        timing='Before comma/period allocations including initial sentinel; synthetic allocation reference only, not human phoneme truth.',
        seed_limit='Two ONNX random nodes; actual saved outputs are identities. No seed-control guarantee.',
        generation_authorized=False,wake_scoring_authorized=False,
        scope='Material/token/observer stage only. A separate pushed generation execution plan and source-role/history admission are required before inference; additional source/negative-background admission and scorer locks before threshold selection.')
    assert len(requests)==132
    with PLAN.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(requests=132,texts=44,inference=False)))


def export():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();p=json.loads(remote)
    for path,h in p['files'].items():assert sha(path)==h
    source=Path('.local/wakeword/source-review-author-jenny-v1/jenny.onnx');model=onnx.load(str(source))
    before=[n.SerializeToString() for n in model.graph.node];weights=[n.SerializeToString() for n in model.graph.initializer]
    model.graph.output.append(onnx.helper.make_tensor_value_info('/Ceil_output_0',onnx.TensorProto.FLOAT,['batch_size',1,'phonemes']))
    assert before==[n.SerializeToString() for n in model.graph.node] and weights==[n.SerializeToString() for n in model.graph.initializer]
    onnx.checker.check_model(model)
    OUT.mkdir(exist_ok=False);path=OUT/'jenny-duration-observer.onnx';onnx.save(model,str(path))
    result=dict(schema='jenny-duration-observer-v1',plan_sha256=sha(PLAN),original_sha256=sha(source),path=str(path),sha256=sha(path),bytes=path.stat().st_size,
        original_graph_nodes_weights_unchanged=True,outputs=[x.name for x in model.graph.output],inference=False,
        limits='Static additive graph output only; no claim of runtime numeric equivalence, token-duration identity, generated quality or source independence.')
    with (RESULTS/'4090-jenny-duration-observer-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','export']);a=p.parse_args()
    plan() if a.action=='plan' else export()
