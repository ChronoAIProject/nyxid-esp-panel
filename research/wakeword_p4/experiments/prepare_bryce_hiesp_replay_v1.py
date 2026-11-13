"""Map every saved Bryce WAV to explicit synthetic paired-development metadata."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qemu_esp32_replay'))
from synthetic_development import REFERENCE, SCHEMA, replay_row, sha, validate_manifest


SOURCE_MANIFEST_SHA = '31adf5d810a50662591da86d1be7a50fbee6042b8e26fc8d091b4ddfb089e909'
GENERATION_PLAN_SHA = '9bcb29aceb08cb3d1fd3c9061b77a1d1c68c51b0b5a4a6477620f3501655f528'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert sha(args.data / 'manifest.json') == SOURCE_MANIFEST_SHA
    assert sha(args.data / 'plan.json') == GENERATION_PLAN_SHA
    source = json.loads((args.data / 'manifest.json').read_text())
    plan = json.loads((args.data / 'plan.json').read_text())
    native = {row['key']: row for row in source['native_records']}
    rows = [replay_row(row, native[row['source_key']]) for row in source['records']]
    assert len(rows) == 126
    assert dict(Counter(r['target'] or 'none' for r in rows)) == {'hey_nyx': 36, 'hi_esp': 36, 'none': 54}
    # All audio stays byte-exact. No synthesis, normalization, timing adjustment,
    # source selection, voice inference or detector scoring occurs here.
    args.output.mkdir(parents=True)
    (args.output / 'wav').mkdir()
    provenance = args.output / 'provenance'
    provenance.mkdir()
    for source_path, name in [(args.data / 'manifest.json', 'source-manifest.json'),
                               (args.data / 'plan.json', 'generation-plan.json'),
                               (args.audit, 'source-audit.json')]:
        shutil.copyfile(source_path, provenance / name)
    for original, derived in zip(source['records'], rows, strict=True):
        path = args.data / original['path']
        assert path.parent.resolve() == args.data.resolve() and sha(path) == original['sha256']
        shutil.copyfile(path, args.output / derived['path'])
    manifest = dict(schema=SCHEMA, role='development-paired-validation', paired_wakephrases=True,
                    source_reference_kind=REFERENCE, independent_final_audio_accessed=False, training=False,
                    source_manifest_path='provenance/source-manifest.json', source_manifest_sha256=SOURCE_MANIFEST_SHA,
                    generation_plan_path='provenance/generation-plan.json', generation_plan_sha256=GENERATION_PLAN_SHA,
                    source_audit_path='provenance/source-audit.json', source_audit_sha256=sha(args.audit),
                    endpoint_protocol=plan['timing_reference'],
                    target_counts={'hey_nyx': 36, 'hi_esp': 36, 'none': 54}, records=rows,
                    scope='Previously consumed synthetic development; generator prefix timing, never human endpoint/final/training labels.')
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    validate_manifest(manifest_path, authenticate_pcm=True)
    report = dict(schema='bryce-hiesp-replay-input-integrity-v1', manifest_sha256=sha(manifest_path),
                  preparer_sha256=sha(Path(__file__)),
                  validator_sha256=sha(Path(__file__).resolve().parents[1] / 'qemu_esp32_replay/synthetic_development.py'),
                  source_manifest_sha256=SOURCE_MANIFEST_SHA, generation_plan_sha256=GENERATION_PLAN_SHA,
                  copied_wavs=126, all_original_wav_and_pcm_hashes_preserved=True,
                  source_order_exact=True, target_counts=manifest['target_counts'],
                  human_phonetic_annotations=False, wake_models_scored=False,
                  qemu_invoked=False, engine_authenticated=False, accepted=None,
                  limits='Source mapping/integrity only. GPU must separately freeze real runner/engine bytes before official replay; no Hi ESP quality, hardware or latency result.')
    (args.output / 'integrity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
