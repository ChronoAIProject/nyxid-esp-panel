"""Apply the preregistered unseen SLR83/dev2-dev3 development gate, no fallback."""
import argparse
import json
from pathlib import Path
from scipy.stats import chi2
from evaluate import event_indices, THRESHOLDS
from synthetic_presentation import load, metrics, sha, trace

MANIFEST = '88cfa446880e8a562927221d552a544bb72f204af071cd6867c7e1284db80415'
INDEX = '865960c9f88fc1d306ab3cfaed0cf3ce861df475375a84d25335263968f24305'
V30 = '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'


def passes(synthetic, upper, minimum):
    return (upper <= .5 and synthetic['equal_voice_condition_macro'] >= minimum
            and synthetic['pre_generation_positive_events'] == 0
            and all(x['false_activation_clips'] == 0 for x in synthetic['negatives'].values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'synthetic', 'baseline', 'background', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh calibration required')
    assert sha(a.manifest) == MANIFEST
    candidate, clips = load(a.synthetic, a.manifest)
    baseline, baseline_clips = load(a.baseline, a.manifest)
    assert baseline['model_sha256'] == V30 and baseline['threshold'] == .91796875
    assert len(clips) == len(baseline_clips) == 90
    assert candidate['official_reports_sha256'] == baseline['official_reports_sha256']
    frozen = metrics(baseline_clips, .91796875)
    minimum = max(.48125, frozen['equal_voice_condition_macro'])
    background = json.loads((a.background/'report.json').read_text())
    assert background['model_sha256'] == candidate['model_sha256']
    assert (background['feature_index_sha256'] == INDEX and not background['independent_final_scored']
            and background['sliding_window'] == 5 and background['split'] == 'validation'
            and background['identity_unit'] == 'recording' and len(background['speakers']) == 86)
    assert len({r['speaker'] for r in background['speakers']}) == 86
    ambient = []
    for item in background['speakers']:
        samples = round(item['source_seconds']*16000)
        assert abs(samples/16000-item['source_seconds']) < 1e-8
        ambient.append(trace(a.background/(item['speaker']+'-scores.npy'), item['score_sha256'], samples))
    hours = sum(r['source_seconds'] for r in background['speakers'])/3600
    assert abs(hours-background['source_hours']) < 1e-10 and abs(hours-8.487218975694445) < 1e-10
    grid = sorted(set(THRESHOLDS+[.5+i/256 for i in range(128)]))
    assert len(grid) == 138 and grid[0] == .5 and grid[-1] == .99609375
    values = []
    for threshold in grid:
        synthetic = metrics(clips, threshold)
        events = sum(len(event_indices(scores, threshold)) for scores in ambient)
        upper = float(.5*chi2.ppf(.95, 2*(events+1))/hours)
        values.append(dict(threshold=threshold,
            synthetic={k: v for k, v in synthetic.items() if k != 'records'},
            background_events=events, background_hours=hours, background_upper_95=upper,
            eligible=passes(synthetic, upper, minimum)))
    eligible = [v for v in values if v['eligible']]
    selected = max(eligible, key=lambda x: (x['synthetic']['equal_voice_condition_macro'],
                   -x['background_upper_95'], x['threshold'])) if eligible else None
    report = dict(role='SLR83 and separate dev2/dev3 development calibration, not final acceptance',
        model_sha256=candidate['model_sha256'], sliding_window=5, validation_only=True,
        threshold=selected['threshold'] if selected else None, selected=selected,
        viable_operating_point=bool(selected), minimum_synthetic_macro=minimum,
        frozen_v30_metrics={k: v for k, v in frozen.items() if k != 'records'}, grid=values,
        manifest_sha256=MANIFEST, source_sha256=sha(Path(__file__)),
        presentation_metrics_source_sha256=sha(Path(__file__).with_name('synthetic_presentation.py')),
        detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
        candidate_report_sha256=sha(a.synthetic/'report.json'),
        baseline_report_sha256=sha(a.baseline/'report.json'),
        background_report_sha256=sha(a.background/'report.json'), final_data_used=False,
        selection='Fixed138 grid; FAR95<=.5/h; zero synthetic negative clips/pre-generation events; equal voice-condition target macro>=max(.48125,new-source frozen-v30 macro). Max macro, lower FAR upper, higher threshold; null if none.',
        limits='Two correlated preset voices, synthetic full-generation/RIR reference only. No merging old background exposure, phonetic truth, human generalization, MCU parity, matched population FAR, or final acceptance claim.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(threshold=report['threshold'], viable_operating_point=bool(selected), minimum_synthetic_macro=minimum)), flush=True)


if __name__ == '__main__':
    main()
