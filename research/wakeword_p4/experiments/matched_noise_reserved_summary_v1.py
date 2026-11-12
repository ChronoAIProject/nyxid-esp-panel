"""Authenticate completed reserved development comparison; never promote a model."""
import hashlib
import json
from pathlib import Path
from scipy.stats import chi2

BASE = Path('.local/wakeword')
OUT = Path('research/wakeword_p4/results')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest_path = BASE/'voxconverse-new-development-features-v1/manifest.json'
manifest = json.loads(manifest_path.read_text())
p = Path('/tmp/nyx-mn-vox-v2/report.json')
official = json.loads(p.read_text())
assert official['manifest_sha256'] == sha(manifest_path)
assert len(official['reports']) == len(manifest['records']) == 86
for raw, row in zip(official['reports'], manifest['records'], strict=True):
    assert raw['record'] == row and raw['pcm_sha256'] == row['pcm_sha256']
    assert raw['input_samples'] == row['samples']
    assert raw['firmware_stage'] == 104 and raw['firmware_crc32_matched']
    assert abs(raw['threshold_requested']-.63) < 1e-6
    assert raw['threshold_getter_after_set'] == .62496001
hours = sum(x['samples'] for x in manifest['records'])/16000/3600
assert abs(hours-official['source_hours']) < 1e-10
assert official['events'] == sum(len(x['events']) for x in official['reports']) == 0
models = {}
inputs = {'official_background': sha(p)}
for version in ['v30', 'matched-noise-independent-formal-v1']:
    bgpath = BASE/f'{version}-voxconverse-new-development-v1/report.json'
    bg = json.loads(bgpath.read_text())
    spath = BASE/f'{version}-slr83-presentation-historical-threshold-v1.json'
    synth = json.loads(spath.read_text())
    assert bg['model_sha256'] == synth['model_sha256']
    assert abs(bg['source_hours']-hours) < 1e-10
    assert len(bg['speakers']) == 86
    for recording, row in zip(bg['speakers'], manifest['records'], strict=True):
        assert recording['speaker'] == row['speaker']
        assert abs(recording['source_seconds']-row['samples']/16000) < 1e-8
    inputs[version] = {'background': sha(bgpath), 'presentation': sha(spath)}
    models[version] = dict(model_sha256=bg['model_sha256'], threshold=synth['threshold'],
        presentation={k:v for k,v in synth['summaries']['nyx'].items()
                      if k not in ('records','voice_conditions')},
        background_events=bg['false_events'], background_hours=hours,
        background_upper95=float(chi2.ppf(.95,2*(bg['false_events']+1))/(2*hours)))
    if version == 'v30':
        models['hiesp'] = dict(threshold_requested=.63, threshold_getter=.62496001,
            presentation={k:v for k,v in synth['summaries']['hiesp'].items()
                          if k not in ('records','voice_conditions')},
            background_events=official['events'], background_hours=hours,
            background_upper95=official['upper_95'])
calpath = BASE/'matched-noise-independent-formal-v1-slr83-new-development-calibration-v1.json'
cal = json.loads(calpath.read_text())
assert len(cal['grid']) == 138 and not any(x['eligible'] for x in cal['grid'])
assert cal['selected'] is None and cal['threshold'] is None
report = dict(script_sha256=sha(Path(__file__)), input_sha256=inputs,
    candidate_calibration_sha256=sha(calpath), models=models,
    candidate_rejected=True, viable_grid_points=0, all86_official_pcm_and_crc_verified=True,
    final_data_used=False, candidate_promoted=False,
    limits='Consumed development only; two correlated preset voices, same full PCM/native frontends '
           'and independently frozen historical thresholds, not matched population FAR. '
           'Generation/RIR references are not human phonetic endpoints or hardware latency. '
           'No new inference, retuning, merging old exposure, final scoring or model promotion.')
for name, raw in [('4090-matched-noise-reserved-development-official-background-v1.json', p.read_bytes()),
                  ('4090-matched-noise-reserved-development-comparison-v1.json',
                   (json.dumps(report,indent=2)+'\n').encode())]:
    target = OUT/name
    assert not target.exists()
    target.write_bytes(raw)
print(json.dumps({k:{x:y for x,y in v.items() if x!='presentation'} for k,v in models.items()}))
