"""Exercise real-source mapping and fail-before-launch admission with fake engines.

The engine/reference fixtures are unit-test bytes, never official inference.
subprocess.Popen is intercepted at the launch boundary; no QEMU is invoked.
"""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch
import wave

import soundfile  # Load the library before intercepting external-launch calls.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qemu_esp32_replay'))
from synthetic_development import sha, validate_manifest


class LaunchBoundaryReached(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'base-runner', 'patched-runner', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    original = args.base_runner.read_text()
    updated = args.patched_runner.read_text()
    unchanged = {}
    for label, begin, end in [
            ('original_final_gates', '    freeze = None\n', "    if row['role'] == 'development-paired-validation':"),
            ('audio_transport_qemu_completion_crc', "    wav = (a.manifest.parent/row['path']).resolve()",
             "    if final and float(parameters[1]) != freeze['threshold_getter_expected']:")]:
        before, after = (text[text.index(begin):text.index(end)] for text in (original, updated))
        assert before == after, label
        unchanged[label] = dict(bytes=len(before.encode()), sha256=hashlib.sha256(before.encode()).hexdigest())
    source_manifest = validate_manifest(args.inputs / 'manifest.json', authenticate_pcm=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix='nyx-synthetic-gate-', dir='/tmp') as temporary:
        work = Path(temporary)
        engines = {}
        for name in ('qemu', 'elf', 'flash'):
            engines[name] = work / ('unit-fixture-' + name)
            engines[name].write_bytes(('Non-executable unit-test ' + name + '; not a real engine.').encode())
        reference = dict(model='wn9_hiesp', sample_rate=16000, chunk_samples=512,
                         firmware_stage=104, firmware_crc32_matched=True,
                         threshold_requested=.63, threshold_getter_after_set=.62496001,
                         **{name + '_sha256': sha(path) for name, path in engines.items()})
        reference_path = work / 'unit-fixture-reference.json'
        reference_path.write_text(json.dumps(reference))
        frozen_path = work / 'unit-fixture-profile.json'
        command = [sys.executable, str(Path(__file__).with_name('freeze_bryce_hiesp_replay_v1.py')),
                   '--manifest', str(args.inputs / 'manifest.json'), '--runner', str(args.patched_runner),
                   '--engine-reference', str(reference_path), '--output', str(frozen_path)]
        for name, path in engines.items():
            command += ['--' + name, str(path)]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        assert json.loads(completed.stdout)['qemu_invoked'] is False
        frozen = json.loads(frozen_path.read_text())
        fixture = work / 'input'
        fixture.mkdir()
        shutil.copytree(args.inputs / 'provenance', fixture / 'provenance')
        (fixture / 'wav').mkdir()
        first = source_manifest['records'][0]
        wav = fixture / first['path']
        shutil.copyfile(args.inputs / first['path'], wav)
        wav_bytes = wav.read_bytes()
        fixture_manifest = fixture / 'manifest.json'
        runner = work / 'run_replay.py'
        runner.write_text(updated)
        base_runner = work / 'base_run_replay.py'
        base_runner.write_text(original)
        profile_path = work / 'active-unit-fixture-profile.json'

        def invoke(label, manifest, profile, use_profile=True, runner_path=runner):
            fixture_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
            profile = {**profile, 'pcm_manifest_sha256': sha(fixture_manifest)}
            profile_path.write_text(json.dumps(profile, indent=2) + '\n')
            output = work / ('output-' + label)
            cli = [str(runner_path), '--manifest', str(fixture_manifest), '--record', manifest['records'][0]['path'],
                   '--threshold', '.63', '--output', str(output), '--nm', str(work / 'unit-fixture-nm')]
            for name, path in engines.items():
                cli += ['--' + name, str(path)]
            if use_profile:
                cli += ['--development-preregistration', str(profile_path)]
            calls = dict(nm=0, launch=0)

            def nm(*_args, **_kwargs):
                calls['nm'] += 1
                return '40081234 D nyx_wakenet_stage\n'

            def launch(*_args, **_kwargs):
                calls['launch'] += 1
                raise LaunchBoundaryReached()

            error, reached = None, False
            with patch.object(sys, 'argv', cli), patch('subprocess.check_output', nm), \
                    patch('subprocess.Popen', launch), patch('sys.stdout', io.StringIO()):
                try:
                    runpy.run_path(str(runner_path), run_name='__main__')
                except LaunchBoundaryReached:
                    reached = True
                except ValueError as rejected:
                    error = str(rejected)
            return dict(reached=reached, error=error, calls=calls,
                        output_created=output.exists(), input_bytes=(output / 'input.pcm').read_bytes()
                        if (output / 'input.pcm').exists() else None)

        accepted = invoke('valid-explicit-development', source_manifest, frozen)
        assert accepted['reached'] and accepted['calls'] == {'nm': 1, 'launch': 1}
        cases.append(dict(name='valid_explicit_development_reaches_intercepted_boundary', passed=True))

        metadata_changes = [
            ('missing_row', lambda m: m['records'].pop()),
            ('reordered_sources', lambda m: m['records'].reverse()),
            ('duplicate_source', lambda m: m['records'].__setitem__(1, deepcopy(m['records'][0]))),
            ('command_end_used_as_keyword_end', lambda m: m['records'][0].__setitem__('keyword_end', m['records'][0]['full_generated_source_end_sample'])),
            ('confuser_relabeled_as_target', lambda m: next(r for r in m['records'] if r['target'] is None).__setitem__('target', 'hey_nyx')),
            ('human_endpoint_claim', lambda m: m['records'][0].__setitem__('human_phonetic_endpoint', True)),
            ('legacy_endpoint_fields', lambda m: m.update(voice_partition_sha256='0' * 64, endpoint_annotation_source_sha256='1' * 64)),
            ('final_header_role', lambda m: m.__setitem__('independent_final_audio_accessed', True)),
            ('final_record_role', lambda m: m['records'][0].update(role='independent-paired-final', split='testing')),
            ('training_role', lambda m: m.__setitem__('training', True)),
            ('changed_condition', lambda m: m['records'][0].__setitem__('condition', 'unknown')),
            ('changed_source_audit_binding', lambda m: m.__setitem__('source_audit_sha256', '0' * 64))]
        for name, change in metadata_changes:
            mutated = deepcopy(source_manifest)
            change(mutated)
            result = invoke(name, mutated, frozen)
            cases.append(dict(name=name, passed=bool(result['error']) and not result['reached']
                              and result['calls'] == {'nm': 0, 'launch': 0} and not result['output_created'],
                              rejection=result['error']))
        profile_changes = [
            ('changed_threshold', lambda p: p.__setitem__('threshold_requested', .62)),
            ('changed_getter', lambda p: p.__setitem__('threshold_getter_expected', .63)),
            ('changed_engine_identity', lambda p: p.__setitem__('qemu_sha256', '0' * 64)),
            ('changed_runner_identity', lambda p: p.__setitem__('runner_sha256', '0' * 64)),
            ('changed_validator_identity', lambda p: p.__setitem__('validator_sha256', '0' * 64)),
            ('profile_final_role', lambda p: p.__setitem__('independent_final_audio_accessed', True)),
            ('changed_frozen_cohort_count', lambda p: p.__setitem__('records', 125))]
        for name, change in profile_changes:
            profile = deepcopy(frozen)
            change(profile)
            result = invoke(name, source_manifest, profile)
            cases.append(dict(name=name, passed=bool(result['error']) and not result['reached']
                              and result['calls'] == {'nm': 0, 'launch': 0} and not result['output_created'],
                              rejection=result['error']))
        for name, manifest in [('profile_omitted', source_manifest),
                               ('profile_omitted_with_legacy_fields', {**source_manifest,
                                'voice_partition_sha256': '0' * 64, 'endpoint_annotation_source_sha256': '1' * 64})]:
            result = invoke(name, manifest, frozen, use_profile=False)
            cases.append(dict(name=name, passed=bool(result['error']) and not result['reached']
                              and result['calls'] == {'nm': 0, 'launch': 0}, rejection=result['error']))
        wav.write_bytes(wav_bytes[:-1] + bytes([wav_bytes[-1] ^ 1]))
        result = invoke('changed_wav_bytes', source_manifest, frozen)
        cases.append(dict(name='changed_wav_bytes', passed=bool(result['error']) and not result['reached']
                          and result['calls'] == {'nm': 0, 'launch': 0}, rejection=result['error']))
        wav.write_bytes(wav_bytes)
        # An unrelated integer-ramp unit WAV tests legacy transport admission;
        # do not relabel any real development source as a transport probe.
        # Both launch calls are intercepted before any engine execution.
        legacy_wav = fixture / 'legacy-unit-fixture.wav'
        legacy_pcm = struct.pack('<1600h', *range(1600))
        with wave.open(str(legacy_wav), 'wb') as stream:
            stream.setparams((1, 2, 16000, 1600, 'NONE', 'not compressed'))
            stream.writeframes(legacy_pcm)
        legacy = dict(records=[dict(path=legacy_wav.name, role='development-transport-probe',
                                   samples=1600, sha256=sha(legacy_wav),
                                   pcm_sha256=hashlib.sha256(legacy_pcm).hexdigest())])
        old_result = invoke('legacy-old', legacy, frozen, use_profile=False, runner_path=base_runner)
        new_result = invoke('legacy-new', legacy, frozen, use_profile=False)
        cases.append(dict(name='legacy_admission_and_transferred_pcm_unchanged',
                          passed=old_result['reached'] and new_result['reached']
                          and old_result['input_bytes'] == new_result['input_bytes']))
    report = dict(schema='bryce-hiesp-replay-contract-audit-v1',
                  auditor_sha256=sha(Path(__file__)), base_runner_sha256=sha(args.base_runner),
                  patched_runner_sha256=sha(args.patched_runner), manifest_sha256=sha(args.inputs / 'manifest.json'),
                  validator_sha256=sha(ROOT / 'qemu_esp32_replay/synthetic_development.py'),
                  original_source_wavs_authenticated=126, unchanged_original_blocks=unchanged,
                  cases=cases, passed=all(r['passed'] for r in cases),
                  engine_fixtures='Non-executable unit-test files and synthetic reference metadata, not actual official engines.',
                  subprocess_launches_intercepted=True, qemu_invoked=False, wake_models_scored=False,
                  actual_gpu_engine_authenticated=False, final_data_accessed=False, accepted=None)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), passed=report['passed'], failed=[r['name'] for r in cases if not r['passed']],
                          qemu_invoked=False, actual_gpu_engine_authenticated=False)))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
