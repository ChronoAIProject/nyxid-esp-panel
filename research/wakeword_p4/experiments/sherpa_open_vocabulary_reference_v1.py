"""Fixed native open-vocabulary reference on existing whole Jenny waveforms only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

R=Path('research/wakeword_p4/results')
BASE=Path('.local/wakeword/sherpa-kws-reference-v1')
MODEL=BASE/'sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01'
LOCK=R/'4090-sherpa-open-vocabulary-reference-plan-v1.json'
OUT=BASE/'jenny-original-v1'
KEYWORDS=BASE/'hey-nicks-keywords.txt'
MODELS={'encoder':MODEL/'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx','decoder':MODEL/'decoder-epoch-12-avg-2-chunk-16-left-64.onnx','joiner':MODEL/'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx'}
MATERIALS={'synthetic':Path('.local/wakeword/jenny-development-replay-v1.json'),'background':Path('.local/wakeword/jenny-earnings-development-v1/manifest.json')}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    import sentencepiece as sp
    archive=BASE/'sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2'
    expected=dict(line.split() for line in (BASE/'checksum.txt').read_text().splitlines())[archive.name]
    assert sha(archive)==expected
    tokenizer=sp.SentencePieceProcessor(model_file=str(MODEL/'bpe.model'))
    pieces=tokenizer.encode('HEY NICKS',out_type=str)
    vocabulary={line.split()[0] for line in (MODEL/'tokens.txt').read_text().splitlines()}
    assert pieces and all(p in vocabulary for p in pieces)
    with KEYWORDS.open('x') as f:f.write(' '.join(pieces)+' @HEY_NICKS\n')
    import sherpa_onnx
    files=[Path(__file__),archive,BASE/'checksum.txt',MODEL/'tokens.txt',MODEL/'bpe.model',KEYWORDS,*MODELS.values(),*MATERIALS.values(),Path(sherpa_onnx.__file__).with_name('keyword_spotter.py')]
    p=dict(schema='sherpa-open-vocabulary-reference-plan-v1',files={str(p):sha(p) for p in files},versions={n:importlib.metadata.version(n) for n in ['sherpa-onnx','sherpa-onnx-core','sentencepiece','numpy','soundfile']},models={k:str(v) for k,v in MODELS.items()},model_total_bytes=sum(p.stat().st_size for p in MODELS.values()),keyword='HEY NICKS',tokens=pieces,keywords_file=str(KEYWORDS),materials={k:str(v) for k,v in MATERIALS.items()},parameters=dict(provider='cpu',num_threads=2,sample_rate=16000,feature_dim=80,max_active_paths=4,keywords_score=1.0,keywords_threshold=.25,num_trailing_blanks=1),protocol=dict(prime_zero_samples=81600,host_input_chunk_samples=480,final_zero_samples=10560,transition_reset=False,native_reset_on_every_detected_keyword=True),retries=0,policy='Native3.3Mparameter ASR-token KWS reference, not another embedding classifier or candidate training. Original public encoderINT8/joinerINT8/decoderFP32, not all-integer TFLite/P4 model. Fixed upstream default keyword settings, no sweeps/aliases/forced alignment/phonetic truth. Existing consumed396Jenny waves+12wholebackgrounds only, full unchanged PCM, no VAD/crop/newTTS/finaldata. Continuous81600prime/native state, ignore prime notifications only; native reset after every detected keyword including prime, never reset at real-input transition. Official-example0.66s finalzeros, all finish events marked separately; only original background duration counts exposure. Host480return clock, native internal batching/lookahead preserved, not same decoder/event policy as v30/HiESP. CPU reference inference only, no training fallback; original wakeword training stays RTX4090. Large native ONNX runtime is not P4 feasibility/acceptance/promotion. Retain v30 and historical failures.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),tokens=pieces,model_total_bytes=p['model_total_bytes'])))
def run():
    import numpy as np
    import soundfile as sf
    import sherpa_onnx
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert importlib.metadata.version(name)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
    OUT.mkdir(exist_ok=False)
    kws=sherpa_onnx.KeywordSpotter(tokens=str(MODEL/'tokens.txt'),keywords_file=p['keywords_file'],**p['models'],**p['parameters'])
    protocol=p['protocol'];prime=protocol['prime_zero_samples'];chunk=protocol['host_input_chunk_samples'];tail=protocol['final_zero_samples'];start=time.monotonic();records=[]
    with (OUT/'events.jsonl').open('x') as log:
        for phase,manifest in p['materials'].items():
            rows=json.loads(Path(manifest).read_text())['records']
            for i,row in enumerate(rows):
                path=row['path'] if phase=='synthetic' else row['wav_path']
                pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
                assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
                stream=kws.create_stream();events=[];fed=0
                # Do not concatenate copies of complete long background recordings.
                for segment in [np.zeros(prime,np.int16),pcm,np.zeros(tail,np.int16)]:
                    for offset in range(0,len(segment),chunk):
                        wave=segment[offset:offset+chunk].astype(np.float32)/32768
                        stream.accept_waveform(rate,wave);fed+=len(wave)
                        while kws.is_ready(stream):
                            kws.decode_stream(stream);keyword=kws.get_result(stream)
                            if keyword:
                                relative=fed-prime
                                events.append(dict(keyword=keyword,return_relative_sample=relative,prime=relative<=0,finish=relative>len(pcm)))
                                kws.reset_stream(stream)
                stream.input_finished()
                while kws.is_ready(stream):
                    kws.decode_stream(stream);keyword=kws.get_result(stream)
                    if keyword:
                        events.append(dict(keyword=keyword,return_relative_sample=fed-prime,prime=False,finish=True));kws.reset_stream(stream)
                rec=dict(phase=phase,index=i,record=row,events=events,real_samples=len(pcm),fed_samples=fed)
                records.append(rec);log.write(json.dumps(rec)+'\n');log.flush()
                print(json.dumps(dict(phase=phase,index=i,events=len(events),elapsed_seconds=time.monotonic()-start)),flush=True)
    assert len(records)==408
    report=dict(schema='sherpa-open-vocabulary-reference-original-v1',plan_sha256=sha(LOCK),records=records,completed=True,elapsed_seconds=time.monotonic()-start,threshold_selected=False,candidate_promoted=False,policy=p['policy'])
    with (OUT/'report.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed=True,inputs=len(records),elapsed_seconds=report['elapsed_seconds'])),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
