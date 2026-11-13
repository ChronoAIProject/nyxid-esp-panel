"""Audit GPU's canonical Joe runner using exact PCM and intercepted launches."""
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

import soundfile  # Resolve native library loading before subprocess interception.


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class InterceptedLaunch(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    original = subprocess.check_output(['git', 'show', 'd5e6f61:research/wakeword_p4/qemu_esp32_replay/run_replay.py'])
    canonical = subprocess.check_output(['git', 'show', '11c658b:research/wakeword_p4/qemu_esp32_replay/run_replay.py'])
    unchanged = {}
    for label, start, end in (
        ('original_final_gates', '    freeze = None\n', "    if row['role'] == 'development-paired-validation':"),
        ('audio_transport_qemu_completion_crc_and_reports', "    wav = (a.manifest.parent/row['path']).resolve()", "\n\nif __name__ == '__main__':")):
        blocks = [blob.decode().split(start, 1)[1].split(end, 1)[0] for blob in (original, canonical)]
        assert blocks[0] == blocks[1]
        unchanged[label] = dict(bytes=len(blocks[0].encode()), sha256=sha(blocks[0].encode()))
    manifest_blob = args.manifest.read_bytes()
    digest = sha(manifest_blob)
    assert digest == 'edf6f7844f654d2a7e22f588873b917381aa8d0957b1e3309a60a0f1973895bc'
    manifest = json.loads(manifest_blob)
    with tempfile.TemporaryDirectory(prefix='joe-policy-', dir='/tmp') as temporary:
        policy = Path(temporary) / 'canonical.py'
        policy.write_bytes(canonical)
        namespace = runpy.run_path(str(policy), run_name='canonical_policy_audit')
    validate = namespace['validate_development_paired']
    for row in manifest['records']:
        validate(manifest, digest, row)
    cases = [dict(name='all360_original_rows_admitted_by_canonical_policy', passed=True)]
    mutations = [
        ('missing_row', lambda m: m['records'].pop()),
        ('reordered_rows', lambda m: m['records'].reverse()),
        ('duplicate_row', lambda m: m['records'].__setitem__(1, deepcopy(m['records'][0]))),
        ('confuser_relabel', lambda m: next(r for r in m['records'] if r['target'] == 'confusion').update(target='hey_nyx')),
        ('wrong_whole_output_end', lambda m: m['records'][0].update(generated_output_end_sample=1)),
        ('wrong_rir_tail', lambda m: next(r for r in m['records'] if r['variant'] == 2).update(rir_tail_samples=0)),
        ('wrong_role', lambda m: m['records'][0].update(role='independent-paired-final', split='testing')),
        ('wrong_voice', lambda m: m['records'][0].update(voice='mike')),
        ('wrong_source_hash', lambda m: m['records'][0].update(source_sha256='0' * 64)),
        ('wrong_end_reference', lambda m: m.update(endpoint_reference='human phonetic endpoint')),
        ('forged_legacy_annotations', lambda m: m.update(voice_partition_sha256='invented', endpoint_annotation_source_sha256='invented')),
        ('final_access', lambda m: m.update(independent_final_audio_accessed=True)),
    ]
    for name, mutate in mutations:
        changed = deepcopy(manifest)
        mutate(changed)
        rejected = False
        try:
            validate(changed, sha(json.dumps(changed).encode()), changed['records'][0])
        except ValueError:
            rejected = True
        assert rejected, name
        cases.append(dict(name=name, passed=True))
    with tempfile.TemporaryDirectory(prefix='joegate-', dir='/tmp') as temporary:
        root = Path(temporary)
        runner = root / 'run_replay.py'
        runner.write_bytes(canonical)
        base_runner = root / 'base_replay.py'
        base_runner.write_bytes(original)
        first = manifest['records'][0]
        shutil.copyfile(args.manifest.parent / first['path'], root / first['path'])
        with wave.open(str(root / first['path']), 'rb') as stream:
            pcm = stream.readframes(stream.getnframes())

        def invoke(name, blob, executable, record):
            path = root / (name + '.json')
            path.write_bytes(blob)
            output = root / name
            argv = [str(executable), '--manifest', str(path), '--record', record, '--output', str(output)]
            for option in ('qemu', 'elf', 'flash', 'nm'):
                argv += ['--' + option, str(root / ('unit-' + option))]
            calls = dict(nm=0, launch=0)

            def nm(*_args, **_kwargs):
                calls['nm'] += 1
                return '40081234 D nyx_wakenet_stage\n'

            def launch(*_args, **_kwargs):
                calls['launch'] += 1
                raise InterceptedLaunch()

            reached, error = False, None
            with patch.object(sys, 'argv', argv), patch('subprocess.check_output', nm), \
                    patch('subprocess.Popen', launch), patch('sys.stdout', io.StringIO()):
                try:
                    runpy.run_path(str(executable), run_name='__main__')
                except InterceptedLaunch:
                    reached = True
                except ValueError as exc:
                    error = str(exc)
            payload = (output / 'input.pcm').read_bytes() if (output / 'input.pcm').exists() else None
            return reached, error, calls, payload, output.exists()

        reached, _, calls, payload, _ = invoke('valid', manifest_blob, runner, first['path'])
        assert reached and calls == {'nm': 1, 'launch': 1}
        assert payload == struct.pack('<IIfI', 0x4e595852, len(pcm) // 2, .63, 0) + pcm
        cases.append(dict(name='exact_whole_selected_pcm_reaches_intercepted_engine_launch', passed=True))
        for name, mutate in mutations:
            changed = deepcopy(manifest)
            mutate(changed)
            reached, error, calls, _, created = invoke(name, json.dumps(changed).encode(), runner, changed['records'][0]['path'])
            assert not reached and error and calls == {'nm': 0, 'launch': 0} and not created, name
        cases.append(dict(name='all12_mutations_reject_in_full_runner_before_nm_output_or_qemu', passed=True))
        ramp = struct.pack('<1600h', *range(1600))
        unit = root / 'legacy-unit.wav'
        with wave.open(str(unit), 'wb') as stream:
            stream.setparams((1, 2, 16000, 1600, 'NONE', 'not compressed'))
            stream.writeframes(ramp)
        legacy = json.dumps(dict(records=[dict(path=unit.name, role='development-transport-probe',
            samples=1600, sha256=sha(unit.read_bytes()), pcm_sha256=sha(ramp))])).encode()
        old = invoke('legacy-old', legacy, base_runner, unit.name)
        new = invoke('legacy-new', legacy, runner, unit.name)
        assert old[0] and new[0] and old[3] == new[3]
        cases.append(dict(name='legacy_unit_admission_and_pcm_transfer_unchanged', passed=True))
    report = dict(schema='mac-canonical-joe-replay-admission-audit-v1', source_sha256=sha(Path(__file__).read_bytes()),
        canonical_commit='11c658b', canonical_runner_sha256=sha(canonical), original_runner_sha256=sha(original),
        manifest_sha256=digest, unchanged_original_blocks=unchanged, cases=cases, passed=True,
        original_rows_admitted=360, full_runner_mutations_rejected_before_launch=12,
        all_launches_intercepted=True, qemu_invoked=False, actual_engine_authenticated=False,
        limits='Admission and exact selected PCM at intercepted launch only. No new or official engine '
               'inference, candidate selection, final audio, source pronunciation or quality claim. '
               'Full scoring orchestration/binary pins remain the GPU executor responsibility.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, original_rows=360, full_runner_rejections=12, qemu_invoked=False)))


if __name__ == '__main__':
    main()
