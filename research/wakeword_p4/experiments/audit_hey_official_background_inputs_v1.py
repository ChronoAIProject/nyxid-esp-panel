"""Authenticate complete development PCM input contracts; no detector calls."""
import hashlib
import json
from pathlib import Path
import zlib

import soundfile as sf

MANIFEST=Path('.local/wakeword/hey-paired-earnings-development-v1/manifest.json')
OUTPUT=Path('research/wakeword_p4/results/4090-hey-official-background-inputs-v1.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()


def audit():
    assert sha(MANIFEST)=='ff0519e735e1c9bb1205ec0ddc7422c7cca74cf1c07678f352b23dca70809841'
    manifest=json.loads(MANIFEST.read_text())
    assert manifest['all_complete'] and manifest['integrity_preparation_passed']
    assert not manifest['acceptance_accessed'] and not manifest['wake_models_scored']
    rows=[]
    for source in manifest['records']:
        path=Path(source['wav_path']);assert sha(path)==source['wav_sha256']
        digest=hashlib.sha256();crc=0;samples=0
        with sf.SoundFile(path) as f:
            assert f.samplerate==16000 and f.channels==1 and f.subtype=='PCM_16'
            while True:
                pcm=f.read(1048576,dtype='int16')
                if not len(pcm):break
                raw=pcm.astype('<i2').tobytes();digest.update(raw);crc=zlib.crc32(raw,crc);samples+=len(pcm)
        assert samples==source['samples'] and digest.hexdigest()==source['pcm_sha256']
        calls=(samples+511)//512
        rows.append(dict(source_id=source['source_id'],samples=samples,pcm_sha256=digest.hexdigest(),
                         pcm_crc32=f'{crc:08x}',input_calls=calls,final_padding_samples=calls*512-samples,
                         exceeds_legacy_one_hour=samples>16000*3600,
                         prime_calls=63,prime_samples=63*512,
                         lifecycle='One fresh engine per whole call; prime separately; no internal reset or crop',
                         final_event_sample_policy='min(input_call*512,original_samples); prime events separate'))
    assert len(rows)==12
    result=dict(schema='hey-official-background-input-contract-v1',manifest_sha256=sha(MANIFEST),
                auditor_sha256=sha(Path(__file__)),records=rows,
                total_samples=sum(r['samples'] for r in rows),maximum_samples=max(r['samples'] for r in rows),
                legacy_one_hour_exceeded=sum(r['exceeds_legacy_one_hour'] for r in rows),
                detector_invoked=False,scoring_authorized=False,acceptance_accessed=False,
                limits='Authenticates Linux complete PCM and transport accounting only; not a runtime/threshold/recognition certification. Mac must compare PCM hashes before same-input claim. No media redistributed.')
    with OUTPUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),maximum_samples=result['maximum_samples'],
                         exceeds_one_hour=result['legacy_one_hour_exceeded'],detector_invoked=False)))


if __name__=='__main__':audit()
