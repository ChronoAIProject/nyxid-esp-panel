"""Whole-wave shared frontend binary with narrow/broad training-only views."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from build_hey_broad_features_v1 import sha, versions, worker, TRAINING

R = Path('research/wakeword_p4/results')
RENDER = Path('.local/wakeword/multispeaker-coverage-rendered-v1')
AMBIENT = Path('.local/wakeword/features-gpu-ami-extra-v1')
OUT = Path('.local/wakeword/multispeaker-coverage-features-v1')
PLAN = R / '4090-multispeaker-coverage-feature-plan-v1.json'


def write_plan():
    import pymicro_features
    import micro_features_cpp
    report = R / '4090-multispeaker-coverage-render-v1.json'
    audit = R / '4090-multispeaker-coverage-render-independent-audit-v1.json'
    scope = R / '4090-multispeaker-render-original-and-audit-exit-v1.json'
    decision = R / '4090-multispeaker-coverage-context-decision-v1.json'
    d, a, s = [json.loads(p.read_text()) for p in [report, audit, scope]]
    assert a['original_report_sha256'] == s['original_report_sha256'] == sha(report)
    assert s['independent_report_sha256'] == sha(audit)
    assert d['cases'] == d['quality_pass'] == a['independently_qualified'] == 23220
    assert not a['failures'] and not s['context_full_source_container_failures']
    frames = json.loads(decision.read_text())['context_frames']
    assert frames == s['actual_context_frames'] == 309
    records = RENDER / 'records.jsonl'
    assert sha(records) == d['records_sha256']
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    files = {str(p): sha(p) for p in [Path(__file__), report, audit, scope, decision, records,
             Path(__file__).with_name('build_hey_broad_features_v1.py'),
             TRAINING / 'features.py', Path(pymicro_features.__file__), Path(micro_features_cpp.__file__),
             Path(__file__).with_name('multispeaker_feature_training_contract_v1.md')]}
    original = json.loads((R / '4090-v30-training.json').read_text())
    ambient_counts = {}
    for name in ['human.json', 'ambient.json']:
        path = AMBIENT / name
        assert sha(path) == original['feature_metadata_sha256'][name]
        files[str(path)] = sha(path)
        selected = [row for row in json.loads(path.read_text()) if row['split'] == 'training']
        ambient_counts[name] = len(selected)
        for row in selected:
            path = AMBIENT / row['path']
            raw = np.load(path, mmap_mode='r', allow_pickle=False)
            assert raw.ndim == 2 and raw.shape[1] == 40 and len(raw) >= frames
            files[str(path)] = sha(path)
    estimated_frames = sum((r['samples'] + 159) // 160 - 2 for r in rows)
    plan = dict(schema='multispeaker-coverage-feature-plan-v1', files=files,
                versions=versions(), workers=8, submission_batch=64, context_frames=frames,
                shared_rows=23220, rows_per_arm=12960, positive_rows_per_arm=2592,
                negative_rows_per_arm=10368, estimated_frames=estimated_frames,
                expected_new_gib=(estimated_frames * 80 + 150 * 1024**2) / 1024**3,
                ambient_training_rows=ambient_counts,
                policy='Unchanged original whole-wave frontend worker; one binary, arm views retain explicit shared offsets. Hard-link binary and original training-only human/ambient arrays. Never extract from cropped waveform or copy protected splits. Bounded64 submissions avoid retaining all results in RAM.',
                boundary='Whole generated positive container including commands, not phonetic wake-prefix. context309/positive jitter3..27. No prefix negative labels inferred from command ends.',
                training_launched=False, detector_invocations=0)
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(estimated_frames=estimated_frames, estimated_binary_bytes=estimated_frames * 80,
                          expected_new_gib=plan['expected_new_gib'], ambient_training_rows=ambient_counts)))


def build():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for path, digest in plan['files'].items():
        assert sha(Path(path)) == digest, path
    assert versions() == plan['versions']
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py',
                    '--expected-gib', str(plan['expected_new_gib'])], check=True)
    OUT.mkdir(exist_ok=False)
    records = [json.loads(line) for line in (RENDER / 'records.jsonl').read_text().splitlines()]
    assert len(records) == plan['shared_rows']
    binary = OUT / 'shared-features.bin'
    common = []
    offset = 0
    with ProcessPoolExecutor(max_workers=plan['workers']) as executor, binary.open('xb') as packed:
        for start in range(0, len(records), plan['submission_batch']):
            chunk = records[start:start + plan['submission_batch']]
            jobs = [(r['order'], r['path'], r['sha256'], r['pcm_sha256'], r['samples']) for r in chunk]
            for (index, raw), r in zip(executor.map(worker, jobs, chunksize=4), chunk, strict=True):
                assert index == r['order'] == len(common) and len(raw) >= plan['context_frames']
                source = r['source']
                assert source['role'] == 'training' and source['label'] == int(source['target'] == 'hey_nyx')
                lead = r['lead_samples'] // 160 - 2
                end = r['generated_prefix_end_sample'] // 160 - 2 if source['label'] else None
                if end is not None:
                    assert end + 27 - plan['context_frames'] <= lead and end + 27 <= len(raw)
                raw.tofile(packed)
                common.append(dict(shared_index=index, split='training', engine=source['engine'],
                    source_id=source['id'], target_id=source['target'], label=source['label'], arms=source['arms'],
                    path=r['path'], sha256=r['sha256'], pcm_sha256=r['pcm_sha256'], samples=r['samples'],
                    offset=offset, frames=len(raw), variant=r['variant'], text=source['text'],
                    lead_frame=lead, keyword_frame=end,
                    generated_container_end_sample=r['generated_container_end_sample'],
                    generated_reference_end_sample=r['generated_prefix_end_sample'],
                    conditioning_id=source['pair_key']['speaker_name'], source_pair_key=source['pair_key']))
                offset += len(raw)
            if len(common) % 1024 == 0 or len(common) == len(records):
                print(json.dumps(dict(rows=len(common), total=len(records), frames=offset)), flush=True)
    assert offset == plan['estimated_frames']
    (OUT / 'shared-metadata.json').write_text(json.dumps(common, separators=(',', ':')) + '\n')
    packs = {}
    for arm in ['narrow', 'broad']:
        root = OUT / arm
        root.mkdir()
        os.link(binary, root / 'features.bin')
        for name in ['human.json', 'ambient.json']:
            selected = [r for r in json.loads((AMBIENT / name).read_text()) if r['split'] == 'training']
            for row in selected:
                dest = root / row['path']
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(AMBIENT / row['path'], dest)
            (root / name).write_text(json.dumps(selected, separators=(',', ':')) + '\n')
        selected = [r for r in common if arm in r['arms']]
        assert len(selected) == 12960 and sum(r['label'] for r in selected) == 2592
        (root / 'features.json').write_text(json.dumps(selected, separators=(',', ':')) + '\n')
        packs[arm] = dict(schema='multispeaker-coverage-target-pack-v1', arm=arm, target_id='hey_nyx',
                         context_frames=309, rows=len(selected), positive_rows=2592, negative_rows=10368,
                         files={name: sha(root / name) for name in ['features.bin', 'features.json', 'human.json', 'ambient.json']},
                         feature_plan_sha256=sha(PLAN), training_only=True)
        (root / 'target-provenance.json').write_text(json.dumps(packs[arm], indent=2) + '\n')
    report = dict(schema='multispeaker-coverage-features-v1', plan_sha256=sha(PLAN),
                  rows=len(common), frames=offset, shared_binary_bytes=binary.stat().st_size,
                  shared_binary_sha256=sha(binary), shared_metadata_sha256=sha(OUT / 'shared-metadata.json'),
                  arms=packs, training_launched=False, detector_invocations=0,
                  limits='Whole-wave feature preparation only; independent binary/role/label/frontend audit required. Shared C++ frontend, no recognition or deployment qualification.')
    with (R / '4090-multispeaker-coverage-features-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rows=len(common), frames=offset, binary_bytes=binary.stat().st_size)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'build'])
    args = parser.parse_args()
    write_plan() if args.action == 'plan' else build()
