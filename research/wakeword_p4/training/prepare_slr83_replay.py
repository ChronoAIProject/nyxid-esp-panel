"""Prepare preregistered full-waveform SLR83 development conditions, no scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = a.dataset/'manifest.jsonl'
    assert sha(manifest) == '8cae0e16afaf97cd82bb8684f4b3aa027f56db834bfbe7a8f5b3911fe66f81fe'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 30
    a.output.mkdir(parents=True, exist_ok=False)
    ir = np.zeros(961, dtype=np.float64)
    ir[[0, 160, 480, 960]] = [1, .25, -.12, .06]
    result = []
    for row in rows:
        assert row['role'] == 'slr83-development-only' and row['quality_pass']
        assert row['split'] == 'validation' and not row['trimmed']
        path = a.dataset/row['path']
        assert sha(path) == row['sha256']
        original, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and original.ndim == 1
        pcm = original.astype(np.float64)/32768
        for variant in range(3):
            speech = fftconvolve(pcm, ir, mode='full') if variant == 2 else pcm.copy()
            lead, tail = 28800, 16000
            wave = np.pad(speech, (lead, tail))
            seed = int(hashlib.sha256((row['path']+f':{variant}:slr83-dev-v1').encode()).hexdigest()[:16], 16)
            snr = None
            if variant:
                snr = 25 if variant == 1 else 15
                rms = float(np.sqrt(np.mean(pcm**2)))
                noise = np.random.default_rng(seed).normal(0, 1, len(wave))
                noise *= rms*10**(-snr/20)/np.sqrt(np.mean(noise**2))
                wave += noise
            peak = float(np.max(np.abs(wave)))
            gain = min(1., .95/max(peak, 1e-8)) if variant else 1.
            wave *= gain
            assert np.isfinite(wave).all() and np.max(np.abs(wave)) < 1
            output = a.output/(Path(row['path']).stem+f'-variant{variant}.wav')
            sf.write(output, wave, 16000, subtype='PCM_16')
            check, rate = sf.read(output, dtype='int16')
            assert rate == 16000 and len(check) == len(wave)
            assert not np.any((check == -32768) | (check == 32767))
            if variant == 0:
                np.testing.assert_array_equal(check[lead:-tail], original)
                assert not np.any(check[:lead]) and not np.any(check[-tail:])
            result.append(dict(path=output.name, role='development-paired-validation',
                split='validation', target=row['target'] if row['target'] in ('hey_nyx', 'hi_esp') else None,
                intended_phrase=row['target'], voice=row['voice'], phonemes=row['phonemes'],
                length_multiplier=row['length_multiplier'], variant=variant, seed=seed, snr_db=snr,
                gain_after_mix=gain, synthetic_rir=variant == 2, rir_tail_samples=960 if variant == 2 else 0,
                source_sha256=row['sha256'], source_samples=len(original), samples=len(check),
                lead_samples=lead, tail_samples=tail, keyword_end=lead+len(speech),
                native_generation_end=lead+len(original),
                keyword_end_method='Full untrimmed generation plus full RIR tail when enabled; not phonetic endpoint',
                sha256=sha(output), pcm_sha256=hashlib.sha256(check.astype('<i2').tobytes()).hexdigest()))
    report = dict(synthesis_protocol='slr83-new-development-v1', records=result,
        paired_wakephrases=True, source_manifest_sha256=sha(manifest),
        script_sha256=sha(Path(__file__)), independent_final_audio_accessed=False,
        wake_scores_computed=False, clean_pcm_preserved_exactly=True,
        rir=dict(delays_samples=[0, 160, 480, 960], amplitudes=[1, .25, -.12, .06]),
        limits='Two development preset voices only; correlated augmentations, no physical-person/global pretraining independence or phonetic truth claim. Source license attribution: Google Inc. 2018/2019, OpenSLR83, CC BY-SA4.0; Demirsahin et al. LREC2020.')
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(records=len(result), manifest_sha256=sha(a.output/'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
