"""Verify existing development read-speech PCM and recount native engine events."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import chi2
from evaluate import event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keyed(rows):
    result = {r['speaker']: r for r in rows}
    require(len(result) == len(rows), 'Duplicate speakers')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'features', 'hiesp-report', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--nyx-report', type=Path, action='append', required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Use fresh output')
    manifest = json.loads(a.manifest.read_text())
    h = json.loads(a.hiesp_report.read_text())
    cache = json.loads((a.features/'cache.json').read_text())
    index = sha(a.features/'human.json')
    require(index == cache['index_sha256'], 'Changed feature index')
    require(sha(a.features/'cache.json') == manifest['source_cache_sha256'], 'Changed source cache')
    features = keyed([r for r in json.loads((a.features/'human.json').read_text()) if r['split'] == 'validation'])
    records = keyed(manifest['records'])
    replay = keyed([dict(r, speaker=r['record']['speaker']) for r in h['reports']])
    require(features.keys() == records.keys() == replay.keys(), 'Incomplete speaker coverage')
    require(h['manifest_sha256'] == sha(a.manifest), 'Changed replay manifest')
    for speaker, row in records.items():
        f, r = features[speaker], replay[speaker]
        require(row['split'] == 'validation' and row['role'] == 'development-read-speech-validation', 'Invalid split/role')
        require(r['record'] == row and r['firmware_stage'] == 104 and r['firmware_crc32_matched'], 'Replay incomplete')
        pcm, rate = sf.read(a.manifest.parent/row['path'], dtype='int16')
        require(rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples'] == r['input_samples'], 'PCM shape mismatch')
        require(sha(a.manifest.parent/row['path']) == row['sha256'], 'Changed WAV')
        require(hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == r['pcm_sha256'] == f['stream_pcm_sha256'], 'Different stream PCM')
        require(sha(a.features/f['path']) == f['feature_sha256'] == row['prior_feature_sha256'], 'Changed feature data')
        require(row['source_seconds'] == f['source_seconds'] == r['source_seconds'], 'Different exposure')
        require(r['calls'] == (len(pcm)+511)//512 and r['final_chunk_zero_pad_samples'] == (-len(pcm)) % 512, 'Replay call/padding mismatch')
    hours = sum(r['source_seconds'] for r in records.values())/3600
    def metrics(count):
        return dict(events=count, source_hours=hours, false_per_hour=count/hours,
                    false_per_hour_upper_95=float(.5*chi2.ppf(.95, 2*(count+1))/hours))
    count = sum(len(r['events']) for r in replay.values())
    require(count == h['false_events'] and abs(hours-h['source_hours']) < 1e-12, 'Hi ESP summary mismatch')
    models = []
    for path in a.nyx_report:
        r = json.loads(path.read_text())
        require(r['split'] == 'validation' and not r['independent_final_scored'] and not r['development_regression_scored'], 'Invalid Nyx evaluation split')
        require(r['feature_index_sha256'] == index and r['feature_cache_sha256'] == sha(a.features/'cache.json'), 'Different Nyx provenance')
        speakers = keyed(r['speakers'])
        require(speakers.keys() == features.keys(), 'Incomplete Nyx coverage')
        total = 0
        for speaker, f in features.items():
            s = speakers[speaker]
            trace = path.parent/(speaker+'-scores.npy')
            require(s['features_sha256'] == f['feature_sha256'] and s['score_sha256'] == sha(trace), 'Changed score provenance')
            scores = np.load(trace, allow_pickle=False)
            require(scores.ndim == 1 and np.isfinite(scores).all(), 'Invalid scores')
            n = len(event_indices(scores, r['threshold']))
            require(n == s['false_events'] and s['source_seconds'] == f['source_seconds'], 'Nyx recount mismatch')
            total += n
        require(total == r['false_events'] and abs(hours-r['source_hours']) < 1e-12, 'Nyx summary mismatch')
        models.append(dict(model_sha256=r['model_sha256'], threshold=r['threshold'], report_sha256=sha(path), **metrics(total)))
    result = dict(role='Same-PCM read-speech development corpus audit, not acceptance',
                  source_sha256=sha(Path(__file__)), manifest_sha256=sha(a.manifest),
                  hiesp_report_sha256=sha(a.hiesp_report), speakers=len(features),
                  exact_stream_pcm_verified=True, feature_and_score_hashes_verified=True,
                  hiesp=dict(threshold_requested=h['threshold_requested'], threshold_getter_values=h['threshold_getter_values'], **metrics(count)),
                  nyx=models, independent_final_audio_accessed=False,
                  limits='Existing validation only; inserted gaps excluded from exposure but all events counted. Native trigger policies and different previously selected thresholds retained. Not matched-general-FAR, positive recall, latency, or independent acceptance evidence.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
