"""Freeze actual existing official replay bytes; never launch or score QEMU."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qemu_esp32_replay'))
from synthetic_development import PREREGISTRATION, REFERENCE, require, sha, validate_manifest, validate_preregistration


PATCHED_RUNNER_SHA = '049643a2d92d8a7cc2b2a13ef3102bdf1bc5fa3fd9517ec31be20ba98ce1b3fe'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'qemu', 'elf', 'flash', 'runner', 'engine-reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    require(sha(args.runner) == PATCHED_RUNNER_SHA, 'Apply/review the exact frozen patch in a separate evaluation worktree')
    manifest = validate_manifest(args.manifest, authenticate_pcm=True)
    require(sha(args.manifest) == 'a14837a2564e6f5bd10aee89e991568bc05827770b750498843a23eecd47b2c2',
            'Only the unchanged126WAV Bryce contract is frozen by this command')
    prior = json.loads(args.engine_reference.read_text())
    prior = prior if isinstance(prior, list) else [prior]
    require(bool(prior), 'Need actual previous official replay reports')
    engine = {name + '_sha256': sha(getattr(args, name)) for name in ('qemu', 'elf', 'flash')}
    for report in prior:
        require(report.get('model') == 'wn9_hiesp' and report.get('sample_rate') == 16000
                and report.get('chunk_samples') == 512 and report.get('firmware_stage') == 104
                and report.get('firmware_crc32_matched') is True
                and report.get('threshold_requested') == .63
                and report.get('threshold_getter_after_set') == .62496001,
                'Reference is not a completed original official fixed-policy replay')
        require(all(report.get(key) == digest for key, digest in engine.items()),
                'Actual engine bytes differ from the completed original reference')
    freeze = dict(schema=PREREGISTRATION, role='development-paired-validation',
                  independent_final_audio_accessed=False, training=False,
                  pcm_manifest_sha256=sha(args.manifest),
                  source_manifest_sha256=manifest['source_manifest_sha256'],
                  generation_plan_sha256=manifest['generation_plan_sha256'],
                  source_audit_sha256=manifest['source_audit_sha256'],
                  source_reference_kind=REFERENCE, records=len(manifest['records']),
                  target_counts=manifest['target_counts'], runner_sha256=sha(args.runner),
                  validator_sha256=sha(Path(__file__).resolve().parents[1] / 'qemu_esp32_replay/synthetic_development.py'),
                  engine_reference_sha256=sha(args.engine_reference), **engine,
                  model='wn9_hiesp', mode='DET_MODE_95', chunk_samples=512, extra_prime_samples=0,
                  threshold_requested=.63, threshold_getter_expected=.62496001,
                  scope='Official historical fixed-policy replay on consumed synthetic development; no acceptance.',
                  execution='Profile creation only. Commit/push and report this exact profile hash before first inference; no automatic runner invocation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(freeze, indent=2) + '\n')
    # Exercise the same admission function the patched runner will call.
    args.development_preregistration = args.output
    args.final_preregistration = None
    args.threshold = .63
    validate_preregistration(args, manifest, args.runner)
    print(json.dumps(dict(profile_sha256=sha(args.output), records=126,
                          engine_bytes_match_completed_reference=True, qemu_invoked=False,
                          wake_models_scored=False)))


if __name__ == '__main__':
    main()
