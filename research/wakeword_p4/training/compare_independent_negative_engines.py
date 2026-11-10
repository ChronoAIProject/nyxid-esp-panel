"""Audit unchanged Nyx and official Hi ESP outcomes on the same independent PCM."""
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


def keyed(rows):
    result = {(r['corpus'], str(r['speaker'])):r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate corpus/speaker')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'hiesp-preregistration', 'hiesp-evaluation', 'nyx-evaluation',
                 'frozen-manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    manifest = json.loads(a.manifest.read_text())
    frozen = json.loads(a.frozen_manifest.read_text())
    prereg = json.loads(a.hiesp_preregistration.read_text())
    hpath, npath = a.hiesp_evaluation/'report.json', a.nyx_evaluation/'report.json'
    h, n = json.loads(hpath.read_text()), json.loads(npath.read_text())
    if (sha(a.manifest) != prereg['pcm_manifest_sha256'] or h['manifest_sha256'] != sha(a.manifest)
            or sha(a.frozen_manifest) != prereg['frozen_negative_manifest_sha256']
            or n['manifest_sha256'] != sha(a.frozen_manifest)
            or h['final_preregistration_sha256'] != sha(a.hiesp_preregistration)
            or n['preregistration_sha256'] != prereg['nyx_preregistration_sha256']
            or not n['independent_negative_final_scored'] or n['threshold_reselected']
            or not h['independent_final_audio_accessed']
            or n['model_sha256'] != prereg['nyx_model_sha256'] or n['threshold'] != prereg['nyx_threshold']
            or h['source_sha256'] != prereg['run_corpus_source_sha256']):
        raise ValueError('Independent engine preregistration/provenance mismatch')
    pcm_rows, prior_rows, nyx_rows = keyed(manifest['records']), keyed(frozen['records']), keyed(n['speakers'])
    official = keyed([dict(r, corpus=r['record']['corpus'], speaker=r['record']['speaker']) for r in h['reports']])
    if (pcm_rows.keys() != prior_rows.keys() or pcm_rows.keys() != nyx_rows.keys()
            or pcm_rows.keys() != official.keys() or len(pcm_rows) != prereg['expected_streams']):
        raise ValueError('Missing independent speaker streams')
    counts = []
    for identity, row in pcm_rows.items():
        old, nr, hr = prior_rows[identity], nyx_rows[identity], official[identity]
        pcm, rate = sf.read(a.manifest.parent/row['path'], dtype='int16')
        if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']
                or sha(a.manifest.parent/row['path']) != row['sha256']
                or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']
                or hr['record'] != row or hr['pcm_sha256'] != row['pcm_sha256']
                or not row['all_frontend_frames_equal'] or not hr['firmware_crc32_matched']
                or hr['firmware_stage'] != 104 or hr['calls'] != (len(pcm)+511)//512
                or hr['final_chunk_zero_pad_samples'] != (-len(pcm)) % 512
                or hr['source_sha256'] != prereg['run_replay_source_sha256']
                or hr['final_preregistration_sha256'] != sha(a.hiesp_preregistration)
                or hr['threshold_requested'] != prereg['threshold_requested']
                or hr['threshold_getter_after_set'] != prereg['threshold_getter_expected']
                or any(hr[k+'_sha256'] != prereg[k+'_sha256'] for k in ('qemu', 'elf', 'flash'))):
            raise ValueError('Same PCM or official runtime completion evidence differs')
        path = a.frozen_manifest.parent/old['path']
        trace = a.nyx_evaluation/(str(old['speaker'])+'-scores.npy')
        if (sha(path) != old['feature_sha256'] or nr['feature_sha256'] != old['feature_sha256']
                or row['prior_feature_sha256'] != old['feature_sha256'] or sha(trace) != nr['score_sha256']):
            raise ValueError('Changed Nyx features or scores')
        scores = np.load(trace, allow_pickle=False)
        indices = event_indices(scores, n['threshold'])
        if [int(i) for i in indices] != nr['event_steps'] or len(indices) != nr['events']:
            raise ValueError('Nyx event recount differs')
        if not row['source_seconds'] == old['source_seconds'] == nr['source_seconds'] == hr['source_seconds']:
            raise ValueError('Different engine exposure')
        counts.append(dict(corpus=identity[0], speaker=identity[1], source_seconds=row['source_seconds'],
                           nyx_events=nr['events'], hiesp_events=len(hr['events'])))
    def metrics(rs, engine):
        count, hours = sum(r[engine+'_events'] for r in rs), sum(r['source_seconds'] for r in rs)/3600
        return dict(events=count, source_hours=hours, false_per_hour=count/hours,
                    false_per_hour_upper_95=float(.5*chi2.ppf(.95, 2*(count+1))/hours))
    pooled = {e:metrics(counts,e) for e in ('nyx','hiesp')}
    if pooled['nyx']['events'] != n['pooled']['events'] or pooled['hiesp']['events'] != h['false_events']:
        raise ValueError('Final corpus summary recount differs')
    result = dict(role='Same-independent-PCM negative fixed-point comparison, not full goal acceptance',
                  source_sha256=sha(Path(__file__)), pcm_manifest_sha256=sha(a.manifest),
                  hiesp_report_sha256=sha(hpath), nyx_report_sha256=sha(npath),
                  hiesp_preregistration_sha256=sha(a.hiesp_preregistration),
                  all_pcm_features_and_scores_verified=True, all_official_replays_complete=True,
                  nyx_model_sha256=n['model_sha256'], nyx_threshold=n['threshold'],
                  hiesp_threshold_requested=h['threshold_requested'],
                  hiesp_threshold_getter=prereg['threshold_getter_expected'],
                  pooled=pooled, by_corpus={c:{e:metrics([r for r in counts if r['corpus']==c],e)
                                             for e in ('nyx','hiesp')} for c in ('test-clean','test-other')},
                  per_stream=counts, independent_negative_final_scored=True,
                  independent_positive_final_scored=False, full_goal_achieved=False,
                  limits='Original engine thresholds, frontends and trigger policies retained. Same independent negative PCM and nominal Poisson bounds only. Equal observed counts do not establish population FAR equivalence. No independent positive recall, phonetic latency, TV/farfield or hardware acceptance claim.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pooled),flush=True)


if __name__ == '__main__':
    main()
