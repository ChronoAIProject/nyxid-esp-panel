"""Training render adapter over the unchanged full-source renderer."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from render_hey_broad_training_v1 import versions

RESULTS = Path('research/wakeword_p4/results')
ORIGINAL = Path(__file__).with_name('render_hey_broad_training_v1.py')
PLAN = RESULTS / '4090-multispeaker-coverage-render-plan-v1.json'
SCHEDULE = Path('.local/wakeword/multispeaker-coverage-render-schedule-v1.jsonl')
OUT = Path('.local/wakeword/multispeaker-coverage-rendered-v1')
NATIVE = Path('.local/wakeword/multispeaker-coverage-native-v1')
DECISION = RESULTS / '4090-multispeaker-coverage-context-decision-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_plan():
    native = NATIVE / 'summary.json'
    audit = RESULTS / '4090-multispeaker-saved-native-independent-audit-v1.json'
    n, a = json.loads(native.read_text()), json.loads(audit.read_text())
    assert a['original_summary_sha256'] == sha(native)
    assert n['pack_qualified'] and a['all_native_process_quality_pass']
    assert n['requested'] == n['quality_passes'] == a['independently_qualified'] == 7740
    records = NATIVE / 'records.jsonl'
    assert sha(records) == n['records_sha256'] == a['records_sha256']
    original_plan = RESULTS / '4090-hey-broad-render-plan-v1.json'
    assert sha(ORIGINAL) == json.loads(original_plan.read_text())['source_sha256']
    rows = []
    maximum_positive = 0
    arm_counts = {arm: 0 for arm in ['narrow', 'broad']}
    for i, r in enumerate(map(json.loads, records.read_text().splitlines())):
        assert r['audit']['passed'] and r['role'] == 'training'
        path = NATIVE / r['path']
        assert sha(path) == r['sha256']
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1 and len(pcm) == r['samples']
        for arm in r['arms']:
            arm_counts[arm] += 1
        if r['label']:
            maximum_positive = max(maximum_positive, len(pcm))
        seeds = [int(hashlib.sha256((r['id'] + ':nyx-coverage-render-v1:' + str(v)).encode()).hexdigest()[:16], 16)
                 for v in range(3)]
        rows.append(dict(order=i, id=r['id'], kind='positive' if r['label'] else 'negative',
                         target=r['target'], label=r['label'], role='training', arms=r['arms'],
                         engine=r['voice'], text=r['text'],
                         pair_key=dict(voice=r['voice'], speaker_name=r['speaker_name'],
                                       speaker_id=r['speaker_id'], scale=r['length_scale'], draw=r['draw']),
                         source_path=str(path), source_sha256=r['sha256'],
                         source_pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                         source_samples=len(pcm), generated_prefix_end=len(pcm) if r['label'] else None,
                         boundary_scope='Whole generated output including commands, not wake-prefix or phonetic endpoint',
                         render_seeds=seeds))
    assert len(rows) == 7740 and arm_counts == dict(narrow=4320, broad=4320)
    # Geometry only, no model responses or holdout observations. Protect the
    # complete positive container because these ONNX models expose no alignment.
    frames = 3 * math.ceil(max(225, math.ceil((maximum_positive + 960) / 160) + 27) / 3)
    lead = max(48000, (frames + 3) * 160)
    decision = dict(schema='multispeaker-coverage-context-decision-v1',
                    native_records_sha256=sha(records), maximum_whole_positive_samples=maximum_positive,
                    context_frames=frames, lead_samples=lead, maximum_jitter_frames=27,
                    rule='Smallest multiple3 >= max(225,ceil((max whole positive output+960 RIR)/160)+27); lead >= context+3frames. Same across both arms, no score/holdout used.',
                    boundary='Whole generated container including command; no native token durations, wake-prefix or phonetic endpoint claim.',
                    limits='Geometry only; exact initialization, stream export, training and resource feasibility still required. No automatic widening of any historical run.')
    with DECISION.open('x') as stream:
        stream.write(json.dumps(decision, indent=2) + '\n')
    with SCHEDULE.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    bound = [Path(__file__), ORIGINAL, original_plan, native, audit, records, DECISION]
    d = dict(schema='multispeaker-coverage-render-plan-v1', source_sha256=sha(ORIGINAL),
             files={str(p): sha(p) for p in bound}, versions=versions(),
             context_frames=frames, lead_samples=lead, tail_samples=16000,
             # The reused renderer authenticates its historical decision as a
             # transitive source receipt only. DECISION above governs this pack.
             context_decision_sha256=sha(RESULTS / '4090-hey-broad-context-decision-v1.json'),
             new_context_decision_path=str(DECISION), new_context_decision_sha256=sha(DECISION),
             schedule=dict(path=str(SCHEDULE), sha256=sha(SCHEDULE), sources=7740, rendered=23220),
             conditions=['quiet', '25dB_noise', '15dB_noise_full_RIR'],
             rir_delays=[0, 160, 480, 960], rir_amplitudes=[1., .25, -.12, .06],
             failure_policy='retain-all-stop-pack-no-substitution', expected_new_gib=10.,
             arm_presentations={arm: count * 3 for arm, count in arm_counts.items()},
             policy='Reuse complete original FFT renderer via explicit AST substitutions of pack identity, total and lead constants only. No trim/respeed; retain complete RIR tail, fixed Gaussian/noise RMS and original noisy peak attenuation. Shared native request rendered once and referenced by both arms.',
             limits='Training only. Equal environment recipes, not identical noise arrays or generated duration across distinct voices; unexposed TTS RNG remains uncontrolled. Whole container boundary only. No development/acceptance/detector/training or promotion authorization.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(d, indent=2) + '\n')
    print(json.dumps(dict(sources=7740, rendered=23220, context_frames=frames, lead_samples=lead, plan_sha256=sha(PLAN))))


def run():
    d = json.loads(PLAN.read_text())
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py',
                    '--expected-gib', str(d['expected_new_gib'])], check=True)
    tree = ast.parse(ORIGINAL.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')

    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            replacements = {'hey-broad-render-result-v1': 'multispeaker-coverage-render-result-v1',
                            '4090-hey-broad-render-v1.json': '4090-multispeaker-coverage-render-v1.json',
                            106956: 23220, 48000: d['lead_samples']}
            if node.value in replacements:
                return ast.copy_location(ast.Constant(replacements[node.value]), node)
            return node

    function = Adapter().visit(function)
    env = dict(__file__=str(ORIGINAL), subprocess=subprocess, json=json, np=np, sf=sf,
               fftconvolve=fftconvolve, Path=Path, PLAN=PLAN, OUT=OUT, RESULTS=RESULTS,
               sha=sha, versions=versions, hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(ORIGINAL), 'exec'), env)
    env['run']()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'run'])
    args = parser.parse_args()
    write_plan() if args.action == 'plan' else run()
