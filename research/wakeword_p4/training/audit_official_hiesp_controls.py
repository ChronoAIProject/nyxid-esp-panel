"""Verify official-fixture transport and expose energy-endpoint limitations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--replay',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    manifest=json.loads(a.manifest.read_text());rows=manifest['records']
    reports=[json.loads(path.read_text()) for path in a.replay]
    by_path={r['record']['path']:r for r in reports}
    if len(by_path)!=len(reports) or set(by_path)!={r['path'] for r in rows}:raise ValueError('Need exact control coverage')
    for key in ('elf_sha256','flash_sha256','qemu_sha256','threshold_requested','threshold_getter_after_set'):
        if len({r[key] for r in reports})!=1:raise ValueError('Mixed engines/thresholds')
    result=[]
    for row in rows:
        path=a.manifest.parent/row['path'];r=by_path[row['path']]
        if (r['record']!=row or r['manifest_sha256']!=sha(a.manifest) or not r['firmware_crc32_matched']
                or r['firmware_stage']!=104 or sha(path)!=row['sha256']):raise ValueError('Changed control/failed replay')
        pcm,rate=sf.read(path,dtype='int16')
        if rate!=16000 or pcm.ndim!=1 or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=r['pcm_sha256']:raise ValueError('Different PCM')
        signal=pcm.astype(np.float32)/32768;count=len(signal)//160
        rms=np.sqrt(np.mean(signal[:count*160].reshape(-1,160)**2,axis=1))
        active=np.flatnonzero(rms>max(1e-6,float(rms.max())*.005))
        if not len(active):raise ValueError('Silent control')
        endpoint=min(len(signal),(int(active[-1])+1)*160+960)/16000
        times=[e['audio_seconds'] for e in r['events']]
        result.append(dict(path=row['path'],pcm_sha256=row['pcm_sha256'],events_seconds=times,
                           emitted_target_control_event=bool(times),energy_proxy_end_seconds=endpoint,
                           event_minus_energy_proxy_ms=[(t-endpoint)*1000 for t in times],
                           proxy_window_events=sum(endpoint-.1<=t<=endpoint+.8 for t in times)))
    report=dict(role='Official Hi ESP fixture transport and endpoint diagnostic; not quality acceptance',
                source_sha256=sha(Path(__file__)),manifest_sha256=sha(a.manifest),replay_sha256=[sha(p) for p in a.replay],
                official_sources=manifest['sources'],records=result,
                independent_final_audio_accessed=False,existing_thresholds_or_windows_changed=False,
                conclusion='Official fixture emits native events in both controls, but existing energy-proxy -100..+800ms window excludes them. Therefore window hits are not equivalent to true target rejection or phonetic latency. Preserve old diagnostic counts; do not use them to rank engines.',
                limits='One official fixture of unknown recording/development provenance, two correlated padding variants. Transport control only, no representative recall, matched FAR, phonetic endpoints or hardware latency. Energy proxy remains a fixed diagnostic, not validated temporal ground truth.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
