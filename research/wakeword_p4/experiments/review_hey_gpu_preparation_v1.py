"""Reproduce Mac's source/metadata-only review of pinned GPU preparation.

Reads git objects and constructed integer boundaries; no media, model inference,
GPU process control, training/development/acceptance access or device I/O.
"""
import argparse
import ast
import hashlib
import json
import subprocess


REF = '35f94d39ef3765ac09d4d22244dfae70fd72cd01'
GEOMETRY = 'research/wakeword_p4/experiments/build_hey_training_features_v1.py'
BACKGROUND = 'research/wakeword_p4/results/4090-hey-earnings-development-complete-preparation-v1.json'


def source(path):
    return subprocess.check_output(['git', 'show', REF + ':' + path])


def geometry():
    code = source(GEOMETRY)
    fn = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'geometry')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'independent-geometry', 'exec'), ns)
    checked = unsupported = 0
    for tail in (0, 960):
        for prefix in range(1, 48001):
            row = dict(id='analytic', role='training', source_sha256='analytic',
                       source_samples=48000, lead_samples=28800, rir_tail_samples=tail)
            timing = dict(id='analytic', role='training', source_waveform_sha256='analytic',
                          allocation_matches_saved_waveform_length=True,
                          generated_prefix_end_16k_sample=prefix)
            geom = ns['geometry'](row, timing)
            if not geom['all_positive_jitters_have_prefix_support']:
                unsupported += 1
                continue
            total_frames = (28800 + 48000 + tail + 16000 + 159) // 160 - 2
            assert geom['keyword_frame'] + 27 <= total_frames
            for jitter in range(3, 28):
                # Unchanged trainer clamp, then frontend raw sample apertures:
                # feature k starts at k*160, finishes at (k+3)*160.
                end = min(total_frames, max(150, geom['keyword_frame'] + jitter))
                assert (end - 150) * 160 <= 28800
                assert (end + 2) * 160 >= 28800 + prefix + tail
                checked += 1
    return dict(schema='mac-independent-hey-feature-geometry-v1', gpu_commit=REF,
        source_path=GEOMETRY, source_sha256=hashlib.sha256(code).hexdigest(),
        tested_generated_prefix_samples_inclusive=[1, 48000], lead_samples=28800,
        rir_tails=[0, 960], context_frames=150, jitters=list(range(3, 28)),
        supported_crop_sample_apertures_checked=checked, unsupported_configurations_preserved=unsupported,
        all_supported_actual_sampler_crops_contain_complete_generated_prefix_and_RIR=True,
        scope='Independent exhaustive integer-boundary geometry with actual trainer clamp and frontend sample-aperture semantics; constructed inputs only, no real new pack, feature inference or training admission.')


def limits():
    data = source(BACKGROUND)
    rows = [dict(source_id=r['source_id'], samples=r['samples'], seconds=r['samples'] / 16000,
                 exceeds_current_native_one_hour_cap=r['samples'] > 16000 * 3600)
            for r in json.loads(data)['records']]
    return dict(schema='mac-native-reference-complete-stream-cap-review-v1', gpu_commit=REF,
        source_manifest_report_sha256=hashlib.sha256(data).hexdigest(), current_native_max_samples=16000 * 3600,
        rows=rows, over_cap=[r['source_id'] for r in rows if r['exceeds_current_native_one_hour_cap']],
        max_samples=max(r['samples'] for r in rows), no_audio_or_scoring_access=True,
        decision='Do not slice recordings or treat rejection as miss. An explicitly versioned opt-in native long-stream contract is required before replay of this complete cohort; current consumed-speech grid remains frozen and unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('geometry', 'stream-limit'))
    args = parser.parse_args()
    print(json.dumps(geometry() if args.command == 'geometry' else limits(), indent=2))
