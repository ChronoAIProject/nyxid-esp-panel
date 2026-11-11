"""Generate only reserved VCTK development voices, never final voice IDs."""
import argparse
import hashlib
import importlib.metadata
import json
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
    assert sha(a.source/'source.json')=='d5617812335a4741d615cc30de40643feeaeb4ebfbd45699d66b222f05294c2f'
    for name,m in source['files'].items(): assert sha(a.source/name)==m['sha256']
    config=json.loads((a.source/'en_GB-vctk-medium.onnx.json').read_text())
    ids=config['phoneme_id_map'];voices=source['partitions']['development']
    assert len(voices)==20 and not set(voices)&set(source['partitions']['reserved_final'])
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
    session=ort.InferenceSession(str(a.source/'en_GB-vctk-medium.onnx'),opt,providers=['CPUExecutionProvider'])
    rows=[]
    with (a.output/'manifest.jsonl').open('x') as manifest:
        for voice in voices:
            for multiplier in (.9,1.,1.1):
                for target,text,phones,tokens in phrases:
                    length_scale=float(config['inference']['length_scale']*multiplier)
                    scales=np.array([config['inference']['noise_scale'],length_scale,config['inference']['noise_w']],np.float32)
                    native=session.run(None,{'input':np.array([tokens],np.int64),
                        'input_lengths':np.array([len(tokens)],np.int64),'scales':scales,
                        'sid':np.array([source['speaker_id_map'][voice]],np.int64)})[0].reshape(-1)
                    pcm=resample_poly(native,320,441)
                    stem=f'{voice}-{multiplier:.1f}-{target}'
                    raw=a.output/(stem+'.npy');np.save(raw,native)
                    wav=a.output/(stem+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16')
                    serialized,rate=sf.read(wav,dtype='int16')
                    finite=bool(np.isfinite(native).all() and np.isfinite(pcm).all())
                    rms=float(np.sqrt(np.mean(pcm**2)));clipped=int(np.count_nonzero(np.abs(pcm)>=1))
                    saturated=int(np.count_nonzero((serialized==-32768)|(serialized==32767)))
                    passed=finite and .2<=len(pcm)/16000<=5 and rms>=.001 and clipped==saturated==0
                    assert rate==16000 and serialized.ndim==1 and len(serialized)==len(pcm)
                    row=dict(role='vctk-development-only',split='validation',voice=voice,speaker_id=source['speaker_id_map'][voice],
                        target=target,text=text,phonemes=phones,token_ids=tokens,scales=scales.tolist(),length_multiplier=multiplier,
                        raw_path=raw.name,raw_sha256=sha(raw),native_rate=22050,native_samples=len(native),
                        path=wav.name,sha256=sha(wav),rate=16000,samples=len(pcm),rms=rms,
                        peak=float(np.max(np.abs(pcm))),clipped_samples=clipped,saturated_pcm16_samples=saturated,
                        quality_pass=passed,trimmed=False,generated_reference_end_sample=len(pcm),
                        native_durations=None,duration_limit='ONNX exposes waveform only; no token durations available')
                    rows.append(row);manifest.write(json.dumps(row)+'\n');manifest.flush()
            print(json.dumps({'voice':voice,'completed':len(rows)}),flush=True)
    report=dict(role='VCTK development synthesis, not final',clips=len(rows),quality_pass=sum(r['quality_pass'] for r in rows),
        source_sha256=sha(a.source/'source.json'),manifest_sha256=sha(a.output/'manifest.jsonl'),script_sha256=sha(Path(__file__)),
        versions={name:importlib.metadata.version(name) for name in ('onnxruntime','piper-phonemize','numpy','scipy','soundfile')},
        voices=voices,final_voices_generated=False,wake_models_scored=False,
        limits='Piper ONNX stochastic synthesis has no exposed seed/durations; preserve raw bytes/hashes rather than claim bit reproducibility. Text-to-phone input verified, not acoustic phonetic truth. Whole untrimmed output end is generation reference. Only20 development voices; reserved89 final untouched.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
