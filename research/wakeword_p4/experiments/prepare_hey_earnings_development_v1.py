"""Prepare all12 preregistered development calls, preserving complete audio.

No wake inference. Acceptance has no executable download/decode path here.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import urllib.request

import numpy as np
from pymicro_features import MicroFrontend
import pymicro_features
import micro_features_cpp
import soundfile as sf

ROOT=Path('research/wakeword_p4/results')
RESERVATION=ROOT/'4090-hey-earnings-background-reservation-v1.json'
PLAN=ROOT/'4090-hey-earnings-development-preparation-plan-v1.json'
FFMPEG=Path('.local/wakeword/ffmpeg-tool-v1/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
OUT=Path('.local/wakeword/hey-paired-earnings-development-v1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def versions():
    return {p:importlib.metadata.version(p) for p in ('numpy','soundfile','pymicro-features')}


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def make_plan():
    files=[Path(__file__),FFMPEG,Path(micro_features_cpp.__file__)]
    files+=sorted(p for p in Path(pymicro_features.__file__).parent.rglob('*')
                  if p.is_file() and p.suffix not in ('.pyc',))
    reservation=json.loads(RESERVATION.read_text())
    result=dict(schema='hey-earnings-development-preparation-plan-v1',reservation_sha256=sha(RESERVATION),
                files={str(p):sha(p) for p in files},versions=versions(),
                decoder_version=subprocess.check_output([str(FFMPEG),'-version'],text=True).splitlines()[0],
                source_ids=[r['source_id'] for r in reservation['records'] if r['role']=='development'],
                aliases=[['hey','willow'],['hey','nyx'],['hey','nix'],['hey','nicks'],['hi','esp'],['hi','e','s','p']],
                decode='Full audio stream0, native float32 duration/hash separately; full mono16k PCM16, CPU flags0/single thread/bitexact/no dither. No offsets/duration limits/gain/silence removal/denoise/segmentation or internal padding.',
                frontend='One persistent MicroFrontend per entire call;10second IO blocks,160sample hops, only final <=159zero pad; raw uint16 features, no model inference.',
                gates='All12 sources/recorded LFS bytes mandatory. Transcript exact alias match stops readiness without replacement. Native decoded duration must match full16k duration within1sample. >=8hours complete decoded exposure; cross-history encoded/PCM overlap audit remains separate.',
                role='development-only',acceptance_accessed=False,wake_models_scored=False,
                limits='Linux decoder differs from Mac9.0.2. Freeze actual binary and PCM hashes; no cross-host identical PCM or physical speaker independence claim.')
    with PLAN.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),sources=len(result['source_ids']),audio_downloaded=False)))


def fetch(url,path,expected_size=None,expected_sha=None):
    part=path.with_suffix(path.suffix+'.partial')
    with urllib.request.urlopen(url,timeout=60) as response,part.open('xb') as f:
        for block in iter(lambda:response.read(1024*1024),b''):f.write(block)
    if expected_size is not None:assert part.stat().st_size==expected_size
    if expected_sha is not None:assert sha(part)==expected_sha
    part.rename(path)


def frontend(path,dest):
    front=MicroFrontend();chunks=[];pcm_hash=hashlib.sha256();samples=0;pad=0
    with sf.SoundFile(path) as source:
        assert source.samplerate==16000 and source.channels==1 and source.subtype=='PCM_16'
        for block in source.blocks(blocksize=160000,dtype='int16'):
            samples+=len(block);pcm_hash.update(block.astype('<i2').tobytes())
            padding=(-len(block))%160
            if padding:pad+=padding;block=np.pad(block,(0,padding))
            for at in range(0,len(block),160):
                frame=front.process_samples(block[at:at+160].tobytes())
                assert frame.samples_read==160
                if frame.features:chunks.append(np.rint(np.asarray(frame.features)*25.6).astype('<u2'))
    assert pad<=159
    array=np.asarray(chunks,dtype='<u2').reshape(-1,40)
    np.save(dest,array,allow_pickle=False)
    return dict(samples=samples,pcm_sha256=pcm_hash.hexdigest(),frames=len(array),final_padding_samples=pad,
                frontend_sha256=sha(dest),array_sha256=hashlib.sha256(array.tobytes()).hexdigest())


def prepare():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();plan=json.loads(remote)
    assert sha(RESERVATION)==plan['reservation_sha256'] and versions()==plan['versions']
    for p,h in plan['files'].items():assert sha(p)==h
    reservation=json.loads(RESERVATION.read_text())
    selected=[r for r in reservation['records'] if r['role']=='development']
    assert [r['source_id'] for r in selected]==plan['source_ids'] and len(selected)==12
    OUT.mkdir(exist_ok=False)
    for name in ('original','transcripts','pcm','features','logs'):(OUT/name).mkdir()
    (OUT/'plan.json').write_bytes(remote)
    rows=[]
    try:
        transcript_reports=[]
        for source in selected:
            path=OUT/'transcripts'/(source['source_id']+'.nlp');fetch(source['transcript_url'],path)
            tokens=[re.sub('[^a-z0-9]','',r['token'].lower()) for r in csv.DictReader(path.open(),delimiter='|')]
            hits=[dict(index=i,alias=alias) for alias in plan['aliases']
                  for i in range(len(tokens)-len(alias)+1) if tokens[i:i+len(alias)]==alias]
            transcript_reports.append(dict(source_id=source['source_id'],sha256=sha(path),tokens=len(tokens),matches=hits))
        dump(OUT/'transcript-audit.json',transcript_reports)
        assert not any(r['matches'] for r in transcript_reports),'Alias match; retain full cohort, do not replace'
        for source in selected:
            identity=source['source_id'];media=OUT/'original'/(identity+'.mp3')
            fetch(source['source_url'],media,source['source_bytes'],source['source_sha256'])
            probe=subprocess.run([str(FFMPEG),'-hide_banner','-i',str(media)],capture_output=True,text=True)
            (OUT/'logs'/(identity+'-probe.txt')).write_text(probe.stderr)
            match=re.search(r'Audio: [^\n]*?, (\d+) Hz, (mono|stereo)',probe.stderr)
            assert match,'Unrecognized native audio metadata'
            rate=int(match[1]);channels=1 if match[2]=='mono' else 2
            assert rate==int(source['metadata']['Sampling Rate (Hz)'])
            common=[str(FFMPEG),'-nostdin','-hide_banner','-cpuflags','0','-threads','1',
                    '-fflags','+bitexact','-i',str(media),'-map','0:a:0','-vn','-threads','1','-flags','+bitexact']
            native_cmd=common+['-c:a','pcm_f32le','-f','f32le','pipe:1']
            native_hash=hashlib.sha256();native_bytes=0
            with (OUT/'logs'/(identity+'-native.txt')).open('wb') as err:
                process=subprocess.Popen(native_cmd,stdout=subprocess.PIPE,stderr=err)
                for data in iter(lambda:process.stdout.read(4*1024*1024),b''):
                    assert len(data)%4==0 and np.isfinite(np.frombuffer(data,dtype='<f4')).all()
                    native_hash.update(data);native_bytes+=len(data)
                assert process.wait()==0
            assert native_bytes%(4*channels)==0
            native_frames=native_bytes//(4*channels)
            wav=OUT/'pcm'/(identity+'.wav')
            pcm_cmd=common+['-ac','1','-ar','16000','-dither_method','none','-c:a','pcm_s16le',str(wav)]
            with (OUT/'logs'/(identity+'-pcm.txt')).open('wb') as err:
                subprocess.run(pcm_cmd,stdout=subprocess.DEVNULL,stderr=err,check=True)
            info=sf.info(wav)
            duration_error=abs(info.frames*rate-native_frames*16000)
            assert duration_error<=rate,'Full decoded duration mismatch'
            features=frontend(wav,OUT/'features'/(identity+'.npy'))
            row=dict(source_id=identity,role='development',source_sha256=sha(media),source_bytes=media.stat().st_size,
                     native_rate=rate,native_channels=channels,native_frames=native_frames,
                     native_float_sha256=native_hash.hexdigest(),native_decode_command=native_cmd,
                     pcm_decode_command=pcm_cmd,wav_sha256=sha(wav),wav_path=str(wav),
                     frontend_path=str(OUT/'features'/(identity+'.npy')),**features)
            rows.append(row);dump(OUT/'progress.json',rows)
            print(json.dumps(dict(prepared=len(rows),source_id=identity,hours=info.frames/16000/3600)),flush=True)
        hours=sum(r['samples'] for r in rows)/16000/3600
        assert len(rows)==12 and hours>=8
        dump(OUT/'manifest.json',dict(records=rows,hours=hours,plan_sha256=sha(PLAN),all_complete=True,
             integrity_preparation_passed=True,overlap_admitted=False,scoring_authorized=False,
             acceptance_accessed=False,wake_models_scored=False))
    except Exception as exc:
        dump(OUT/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',completed=len(rows),
             replacement=False,wake_models_scored=False,acceptance_accessed=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('write-plan','prepare'))
    a=p.parse_args()
    make_plan() if a.action=='write-plan' else prepare()
