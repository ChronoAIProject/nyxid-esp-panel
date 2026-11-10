"""Generate paired calibration-only wake phrases from existing validation voices.

Never reads or modifies the independent-final corpus. Both wake phrases share
voice/speed/rendering; words remain different, so this is a paired comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly
from comparison_holdout import Session
from kokoro_samples import MODEL_SHA, VOICES_SHA


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('assets','voice-partitions','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use fresh output')
    for name,expected in [('kokoro-v1.0.onnx',MODEL_SHA),('voices-v1.0.bin',VOICES_SHA)]:
        if sha(a.assets/name)!=expected:raise ValueError('Changed TTS asset')
    partitions=json.loads(a.voice_partitions.read_text())['voice_partitions']
    voices=sorted(partitions['validation'])
    if (len(voices)!=4 or set(voices)&(set(partitions['training'])|set(partitions['testing'])) or
            any(not v.startswith(('af_','am_','bf_','bm_')) for v in voices)):
        raise ValueError('Require four existing disjoint English validation presets')
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=Session(str(a.assets/'kokoro-v1.0.onnx'),opts,providers=['CPUExecutionProvider'])
    engine=Kokoro.from_session(session,str(a.assets/'voices-v1.0.bin'))
    plan=[('Hey Nicks.','hey_nyx'),('Hi E S P.','hi_esp'),('Hey Nick.',None),
          ('Hey Next.',None),('Hey Mix.',None),('Please check the volume.',None)]
    a.output.mkdir(parents=True);records=[]
    for voice in voices:
        lang='en-gb' if voice.startswith('b') else 'en-us'
        for speed in (.9,1.,1.1):
            session.requested_speed=speed
            for i,(text,target) in enumerate(plan):
                signal,rate=engine.create(text,voice=voice,speed=speed,lang=lang)
                if rate!=24000:raise ValueError('Expected24kHz TTS')
                signal=resample_poly(signal,2,3)
                pcm=np.clip(np.rint(np.pad(signal,(28800,16000))*32768),-32768,32767).astype('<i2')
                path=Path(voice)/f'{speed:.1f}-{i}.wav';(a.output/path).parent.mkdir(parents=True,exist_ok=True)
                sf.write(a.output/path,pcm,16000,subtype='PCM_16')
                decoded,rate=sf.read(a.output/path,dtype='int16');np.testing.assert_array_equal(pcm,decoded)
                records.append(dict(path=str(path),role='development-paired-validation',split='validation',
                                    text=text,target=target,voice=voice,speed=speed,samples=len(pcm),
                                    lead_samples=28800,synthesis_end_samples=28800+len(signal),
                                    sha256=sha(a.output/path),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                    phonemes=engine.tokenizer.phonemize(text,lang)))
        print(json.dumps(dict(voice=voice,clips=len(records))),flush=True)
    report=dict(role='Paired existing-voice development validation, never final',
                paired_wakephrases=True,voice_ids=voices,voice_partition_sha256=sha(a.voice_partitions),
                model_sha256=MODEL_SHA,voices_sha256=VOICES_SHA,source_sha256=sha(Path(__file__)),
                records=records,independent_final_audio_accessed=False,
                protocol='4 existing validation voices x3 speeds x2 target phrases and4 common negative prompts. Same rendering;16k mono PCM16 with1.8s lead and1s tail. Reset between clips.',
                limits='Repeated development synthetic speech, not physical-speaker independence. Synthesis end includes possible TTS silence, not a phonetic endpoint. ASR/pronunciation consistency screen required. This corpus cannot establish representative product recall.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    generation={k:v for k,v in report.items() if k!='records'}
    generation.update(clips=len(records),manifest_sha256=sha(a.output/'manifest.jsonl'),
                      json_manifest_sha256=sha(a.output/'manifest.json'),scores_computed=False)
    (a.output/'generation.json').write_text(json.dumps(generation,indent=2)+'\n');print(json.dumps(generation))


if __name__=='__main__':main()
