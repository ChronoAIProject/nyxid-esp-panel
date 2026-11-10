"""Add fixed lead/tail silence to all frozen paired WAVs without detector inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    original = a.source/'manifest.jsonl'
    expected = '40b4ed9e88a4c8c7d7a1b5d6ec94522c15cfc39a91ed763ffcadd996d2f468eb'
    generation = json.loads((a.source/'generation.json').read_text())
    if sha(original) != expected or generation['manifest_sha256'] != expected:
        raise ValueError('Changed original independent paired challenge')
    rows = [json.loads(line) for line in original.read_text().splitlines()]
    counts = Counter(str(r['target']) for r in rows)
    if counts != {'hey_nyx':78, 'hi_esp':78, 'None':312} or len({r['path'] for r in rows}) != 468:
        raise ValueError('Changed frozen target population')
    if any(r['voice'].startswith(('af_', 'am_', 'bf_', 'bm_')) for r in rows):
        raise ValueError('Development English preset in final challenge')
    a.output.mkdir(parents=True)
    records = []
    for row in rows:
        source = (a.source/row['path']).resolve()
        if not source.is_relative_to(a.source.resolve()) or sha(source) != row['sha256']:
            raise ValueError('Changed or escaped original WAV')
        info = sf.info(source)
        pcm, rate = sf.read(source, dtype='int16')
        if rate != 16000 or info.subtype != 'PCM_16' or pcm.ndim != 1 or len(pcm) != row['samples']:
            raise ValueError('Changed original PCM shape/format')
        signal = np.pad(pcm, (28800,16000))
        target = a.output/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        sf.write(target,signal,16000,subtype='PCM_16')
        decoded, rate = sf.read(target,dtype='int16')
        np.testing.assert_array_equal(signal,decoded)
        records.append(dict(row, role='independent-paired-final', split='testing',
                            source_wav_sha256=row['sha256'],
                            source_pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                            source_samples=len(pcm), lead_samples=28800, tail_samples=16000,
                            samples=len(signal), sha256=sha(target),
                            pcm_sha256=hashlib.sha256(signal.astype('<i2').tobytes()).hexdigest()))
    result = dict(role='Frozen independent paired waveform preparation; no detector scores',
                  source_sha256=sha(Path(__file__)), source_holdout_manifest_sha256=sha(original),
                  source_generation_sha256=sha(a.source/'generation.json'),
                  paired_wakephrases=True, records=records, counts=dict(counts),
                  independent_positive_final_audio_accessed_for_preparation=True,
                  detector_scores_computed=False, records_removed_or_relabeled=0,
                  limits='Original TTS PCM retained exactly with fixed1.8s lead and1s tail. Text/target are synthesis intent, not independent pronunciation truth. No phonetic endpoint labels; no recall or latency acceptance conclusion.')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
