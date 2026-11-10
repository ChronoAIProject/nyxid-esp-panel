"""Same decoded developer PCM through Hey Nyx and actual emulated Hi ESP.

This demonstrates plumbing only, not a matched-FAR quality comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from features import features
from evaluate import Detector, event_indices, smooth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'nyx-model', 'nyx-calibration', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--hiesp-report', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use fresh output')
    manifest = json.loads(a.manifest.read_text())
    calibration = json.loads(a.nyx_calibration.read_text())
    if sha(a.nyx_model) != calibration['model_sha256'] or not calibration.get('validation_only'):
        raise ValueError('Need unchanged model and prior validation-only threshold')
    threshold, window = calibration['threshold'], calibration['sliding_window']
    reports = [json.loads(path.read_text()) for path in a.hiesp_report]
    by_path = {r['record']['path']:r for r in reports}
    if len(by_path) != len(reports) or set(by_path) != {r['path'] for r in manifest['records']}:
        raise ValueError('Hi ESP reports must exactly cover manifest')
    detector = Detector(a.nyx_model)
    rows = []
    for row in manifest['records']:
        if row['role'] != 'development-transport-probe':
            raise ValueError('Only developer probes allowed')
        path = a.manifest.parent/row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Changed WAV')
        pcm, rate = sf.read(path, dtype='int16'); raw = pcm.astype('<i2').tobytes()
        other = by_path[row['path']]
        if (rate != 16000 or hashlib.sha256(raw).hexdigest() != row['pcm_sha256'] or
                other['pcm_sha256'] != row['pcm_sha256'] or not other['firmware_crc32_matched'] or
                other['firmware_stage'] != 104 or other['manifest_sha256'] != sha(a.manifest)):
            raise ValueError('Same-PCM or firmware provenance mismatch')
        data = features(pcm)
        detector.reset(); scores = detector.predict(data)
        detector.reset(); np.testing.assert_array_equal(scores, detector.predict(data))
        indices = event_indices(smooth(scores, window), threshold)
        rows.append(dict(path=row['path'], text=row['text'], voice=row['voice'],
                         pcm_sha256=row['pcm_sha256'], nyx_events_seconds=[float((3*(i+window-1)+5)*.01) for i in indices],
                         hiesp_events=other['events'], nyx_reset_repeatable=True))
    result = dict(role='Same-PCM developer engine transport comparison, not acceptance',
                  source_sha256=sha(Path(__file__)), manifest_sha256=sha(a.manifest),
                  nyx_model_sha256=sha(a.nyx_model), nyx_validation_calibration_sha256=sha(a.nyx_calibration),
                  nyx_threshold=threshold, nyx_smoothing_window=window,
                  hiesp_report_sha256=[sha(p) for p in a.hiesp_report],
                  hiesp_requested_thresholds=sorted({r['threshold_requested'] for r in reports}),
                  hiesp_getter_thresholds=sorted({r['threshold_getter_after_set'] for r in reports}),
                  records=rows, independent_final_audio_accessed=False,
                  limits='Six training-voice TTS developer probes. Both engines saw same original PCM; each retains its own frontend/state policy. Thresholds are NOT matched by false-wake rate. Synthetic end timestamps not phonetic alignment. No performance superiority, representative recall/FAR or hardware latency conclusion.')
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__ == '__main__':
    main()
