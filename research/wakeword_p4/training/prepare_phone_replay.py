"""Freeze padded paired development PCM with explicit generation-end reference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--partitions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',choices=['kokoro','vctk'],default='kokoro')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=a.dataset/'manifest.jsonl'
    expected={'kokoro':'10b5e060163f2eaf896ff8219cd9c1ff341c569f4e585b3a68828fb71bef2b94',
              'vctk':'dcf1bbd2ee183bd1bdb893f4662e94d74cc6554b2cc15651834bdc5e59f030da'}
    assert sha(manifest)==expected[a.protocol]
    source=[json.loads(x) for x in manifest.read_text().splitlines()]
    metadata=json.loads(a.partitions.read_text())
    if a.protocol=='kokoro':
        partitions=metadata['voice_partitions']
        role='development-only-explicit-phone-paired'
    else:
        assert sha(a.partitions)=='d5617812335a4741d615cc30de40643feeaeb4ebfbd45699d66b222f05294c2f'
        partitions={'validation':metadata['partitions']['development'], 'training':[],
                    'testing':metadata['partitions']['reserved_final']}
        role='vctk-development-only'
    rows=[]
    for r in source:
        assert r['role']==role and r['split']=='validation' and r['quality_pass']
        assert r['voice'] in partitions['validation'] and r['voice'] not in partitions['training']+partitions['testing']
        assert sha(a.dataset/r['path'])==r['sha256']
        pcm,rate=sf.read(a.dataset/r['path'],dtype='int16')
        assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples']
        wave=np.pad(pcm,(28800,16000));path=a.output/r['path'];sf.write(path,wave,16000,subtype='PCM_16')
        check,_=sf.read(path,dtype='int16');np.testing.assert_array_equal(check[28800:-16000],pcm)
        rows.append(dict(path=path.name,role='development-paired-validation',split='validation',
            target=r['target'] if r['target'] in ('hey_nyx','hi_esp') else None,
            intended_phrase=r['target'],voice=r['voice'],speed=r.get('speed'),length_multiplier=r.get('length_multiplier'),phonemes=r['phonemes'],
            sha256=sha(path),pcm_sha256=hashlib.sha256(wave.astype('<i2').tobytes()).hexdigest(),samples=len(wave),
            source_sha256=r['sha256'],source_samples=len(pcm),lead_samples=28800,tail_samples=16000,
            keyword_end=28800+len(pcm),keyword_end_method='Full untrimmed generation end; not final-phone boundary'))
    result=dict(synthesis_protocol=a.protocol,records=rows,paired_wakephrases=True,voice_partition_sha256=sha(a.partitions),
        endpoint_annotation_source_sha256=sha(Path(__file__)),source_manifest_sha256=sha(manifest),
        independent_final_audio_accessed=False,wake_scores_computed=False,
        endpoint_protocol='Whole untrimmed generation end with fixed -100..+800ms diagnostic window; full source PCM preserved. No claim of phonetic boundary accuracy.',
        limits=('Historical Kokoro development voices reused; not new independent voices.' if a.protocol=='kokoro' else 'VCTK development voices only; reserved final voices excluded; same Piper architecture as training generator, different checkpoint/data.')+' No final data.')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(records=len(rows),manifest_sha256=sha(a.output/'manifest.json'))))


if __name__=='__main__': main()
