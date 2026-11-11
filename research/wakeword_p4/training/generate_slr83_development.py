"""Generate two pinned OpenSLR83 development voices, never training/final."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
import onnxruntime as ort
from piper_phonemize import phonemize_espeak
from scipy.signal import resample_poly
import soundfile as sf


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.source/'source.json').read_text())
    assert sha(a.source/'source.json')=='10f1c7b0351c6349fb200245b4ccb8a5cd68d929bd0759cfbc911838d51431d8'
    for name,m in source['files'].items(): assert sha(a.source/name)==m['sha256']
    voices=sorted(source['voices'])
    assert len(voices)==2
    rows=[]
    with (a.output/'manifest.jsonl').open('x') as manifest:
        for voice in voices:
            model_name=f'en_GB-{voice}-{source["voices"][voice]}.onnx'
            config=json.loads((a.source/voice/(model_name+'.json')).read_text())
            ids=config['phoneme_id_map'];assert config['num_speakers']==1
            native_rate=config['audio']['sample_rate']
            plan=[('hey_nyx','Hey Nicks.'),('hi_esp','Hi E S P.'),('hey_nick','Hey Nick.'),
                  ('hey_next','Hey Next.'),('hey_mix','Hey Mix.')]
            phrases=[]
            for target,text in plan:
                sentences=phonemize_espeak(text,config['espeak']['voice']);assert len(sentences)==1
                phones=sentences[0];assert all(phone in ids for phone in phones)
                tokens=list(ids['^'])+list(ids['_'])
                for phone in phones: tokens+=list(ids[phone])+list(ids['_'])
                tokens+=list(ids['$'])
                phrases.append((target,text,phones,tokens))
            opt=ort.SessionOptions();opt.intra_op_num_threads=3;opt.inter_op_num_threads=1
            session=ort.InferenceSession(str(a.source/voice/model_name),opt,providers=['CPUExecutionProvider'])
            for multiplier in (.85,1.,1.15):
                for target,text,phones,tokens in phrases:
                    length_scale=float(config['inference']['length_scale']*multiplier)
                    scales=np.array([config['inference']['noise_scale'],length_scale,config['inference']['noise_w']],np.float32)
                    native=session.run(None,{'input':np.array([tokens],np.int64),
                        'input_lengths':np.array([len(tokens)],np.int64),'scales':scales})[0].reshape(-1)
                    divisor=math.gcd(16000,native_rate)
                    pcm=resample_poly(native,16000//divisor,native_rate//divisor) if native_rate!=16000 else native.copy()
                    stem=f'{voice}-{multiplier:.2f}-{target}'
                    raw=a.output/(stem+'.npy');np.save(raw,native)
                    wav=a.output/(stem+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16')
                    serialized,rate=sf.read(wav,dtype='int16')
                    finite=bool(np.isfinite(native).all() and np.isfinite(pcm).all())
                    rms=float(np.sqrt(np.mean(pcm**2)));clipped=int(np.count_nonzero(np.abs(pcm)>=1))
                    saturated=int(np.count_nonzero((serialized==-32768)|(serialized==32767)))
                    passed=finite and .2<=len(pcm)/16000<=5 and rms>=.001 and clipped==saturated==0
                    assert rate==16000 and serialized.ndim==1 and len(serialized)==len(pcm)
                    row=dict(role='slr83-development-only',split='validation',voice=voice,speaker_id=0,
                        target=target,text=text,phonemes=phones,token_ids=tokens,scales=scales.tolist(),length_multiplier=multiplier,
                        raw_path=raw.name,raw_sha256=sha(raw),native_rate=native_rate,native_samples=len(native),
                        path=wav.name,sha256=sha(wav),rate=16000,samples=len(pcm),rms=rms,
                        peak=float(np.max(np.abs(pcm))),clipped_samples=clipped,saturated_pcm16_samples=saturated,
                        quality_pass=passed,trimmed=False,generated_reference_end_sample=len(pcm),
                        native_durations=None,duration_limit='ONNX exposes waveform only; no token durations available')
                    rows.append(row);manifest.write(json.dumps(row)+'\n');manifest.flush()
            print(json.dumps({'voice':voice,'completed':len(rows)}),flush=True)
    report=dict(role='OpenSLR83 development synthesis, not training/final',clips=len(rows),quality_pass=sum(r['quality_pass'] for r in rows),
        source_sha256=sha(a.source/'source.json'),manifest_sha256=sha(a.output/'manifest.jsonl'),script_sha256=sha(Path(__file__)),
        versions={name:importlib.metadata.version(name) for name in ('onnxruntime','piper-phonemize','numpy','scipy','soundfile')},
        voices=voices,final_voices_generated=False,wake_models_scored=False,
        limits='Piper ONNX stochastic synthesis has no exposed seed/durations; preserve raw bytes/hashes rather than claim bit reproducibility. Text-to-phone input verified, not acoustic phonetic truth. Whole untrimmed output end is generation reference. Two development preset voices only; no certified physical-person/pretraining disjointness. VCTK reserved89 final untouched.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
