"""Add exact500ms zero lead for detector-independent alignment stability tests."""
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
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    if manifest['independent_final_audio_accessed'] or manifest['wake_scores_accessed']:
        raise ValueError('Need detector-independent development inputs')
    a.output.mkdir(parents=True);rows=[]
    for row in manifest['records']:
        path=(a.inputs/'corpus'/row['path']).resolve()
        if not path.is_relative_to((a.inputs/'corpus').resolve()) or sha(path)!=row['sha256']:
            raise ValueError('Changed/escaped source waveform')
        pcm,rate=sf.read(path,dtype='int16')
        if rate!=16000 or pcm.ndim!=1 or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=row['pcm_sha256']:
            raise ValueError('Changed PCM')
        shifted=np.pad(pcm,(8000,0)).astype('<i2');target=a.output/'corpus'/row['path']
        target.parent.mkdir(parents=True,exist_ok=True);sf.write(target,shifted,16000,subtype='PCM_16')
        decoded,rate=sf.read(target,dtype='int16');np.testing.assert_array_equal(decoded[8000:],pcm)
        if np.any(decoded[:8000]) or rate!=16000:raise ValueError('Unexpected leading samples')
        target.with_suffix('.lab').write_text(row['transcript']+'\n')
        rows.append(dict(row,sha256=sha(target),pcm_sha256=hashlib.sha256(shifted.tobytes()).hexdigest(),
                         samples=len(shifted),parent_wav_sha256=row['sha256'],parent_pcm_sha256=row['pcm_sha256'],added_lead_samples=8000))
    result=dict(manifest,role='500ms pure translation automatic-alignment development controls',records=rows,
                source_sha256=sha(Path(__file__)),parent_input_manifest_sha256=sha(a.inputs/'manifest.json'),
                added_lead_seconds=.5,remaining_original_pcm_verified=True,
                prospective_test='Same fixed MFA models/config;compare word and terminal-phone boundaries after subtracting500ms. Endpoint stability tolerance20ms, report every failure without changing labels/windows.',
                limits='Timing equivariance is necessary evidence for alignment consistency, not proof of phonetic accuracy. Repeated identical waveform bodies are not independent speech. No wake events, acceptance scores or final audio.')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(files=len(rows),added_lead_samples=8000,remaining_original_pcm_verified=True)))


if __name__=='__main__':main()
