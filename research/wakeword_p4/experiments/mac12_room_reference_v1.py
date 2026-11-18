"""Paired energy-matched dry and measured-room Mac witnesses at frozen operating points."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib

os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path('research/wakeword_p4')
RESULTS = ROOT / 'results'
BASE = Path('.local/wakeword/mac12-room-reference-v1')
PLAN = RESULTS / '4090-mac12-room-reference-plan-v1.json'
ARTIFACT = RESULTS / '4090-mac12-room-reference-evidence-v1.tar.gz'
REPORT = RESULTS / '4090-mac12-room-reference-original-v1.json'
QEMU = ROOT / 'qemu_hey_reference_v2'
sys.path.insert(0, str(QEMU.resolve()))
import run_willow_mac_matched_prime_v1 as official
from run_c161_binary40_parallel_v1 import MODELS, dual
from collect_multispeaker_formal_v1 import prepare
from uint8_event_policy_v1 import event_steps


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def plan():
    import importlib.metadata as metadata

    prior_path = RESULTS / '4090-willow-mac-matched-prime-plan-v1.json'
    parents_path = RESULTS / '4090-c161-binary40-parallel-runtime-plan-v1.json'
    rooms_path = RESULTS / '4090-mit-rir-audit.json'
    prior, parents, rooms = map(read, (prior_path, parents_path, rooms_path))
    files = {path: prior['files'][path] for path in prior['runtime'].values()}
    files.update({path: parents['files'][path] for path in MODELS.values()})
    for path in (Path(__file__), Path(official.__file__), official.SOURCE, prior_path, parents_path,
                 rooms_path, ROOT / 'training/features.py', ROOT / 'experiments/run_c161_binary40_parallel_v1.py',
                 ROOT / 'experiments/collect_multispeaker_formal_v1.py', ROOT / 'experiments/uint8_event_policy_v1.py'):
        files[str(path)] = sha(path)

    # First training-room entry by path, chosen independently of recognition results.
    room = min((r for r in rooms['records'] if r['split'] == 'training'), key=lambda r: r['path'])
    rir = Path('.local/wakeword/mit-rirs') / room['path']
    files[str(rir)] = room['sha256']
    for path, digest in files.items():
        assert sha(path) == digest, path
    response, rate = sf.read(rir, dtype='float64')
    assert rate == room['rate'] == 16000 and response.shape == (room['samples'],)
    assert np.isfinite(response).all() and np.linalg.norm(response) > 0

    folder = BASE / 'render'
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    for i, source in enumerate(prior['records']):
        original, rate = sf.read(source['path'], dtype='int16')
        assert sha(source['path']) == source['sha256'] and rate == 16000
        assert original.shape == (source['samples'],)
        assert hashlib.sha256(original.astype('<i2').tobytes()).hexdigest() == source['pcm_sha256']
        files[source['path']] = source['sha256']
        original = original.astype(np.float64)
        direct = np.pad(original, (0, len(response) - 1))
        wet = fftconvolve(original, response, mode='full')
        wet_gain = float(np.linalg.norm(direct) / np.linalg.norm(wet))
        wet *= wet_gain
        common_gain = min(1.0, 32767.0 / max(np.abs(direct).max(), np.abs(wet).max()))
        energy = []
        for condition, wave in [('direct', direct), ('bedroom', wet)]:
            rounded = np.rint(wave * common_gain)
            assert np.isfinite(rounded).all() and np.abs(rounded).max() <= 32767
            pcm = rounded.astype('<i2')
            path = folder / f'{i:02}-{condition}.wav'
            sf.write(path, pcm, rate, subtype='PCM_16')
            energy.append(float(np.linalg.norm(pcm.astype(np.float64))))
            records.append(dict(index=i, condition=condition, voice=source['voice'], form=source['form'],
                                target=source['target'], text=source['text'], path=str(path), sha256=sha(path),
                                pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), samples=len(pcm),
                                original_pcm_sha256=source['pcm_sha256'], room_energy_gain=wet_gain,
                                common_peak_gain=common_gain, pcm_l2=energy[-1]))
        assert abs(20 * np.log10(energy[1]/energy[0])) < 0.01
    assert len(records) == 24
    versions = {name: metadata.version(name) for name in [*parents['versions'], 'scipy']}
    write_new(PLAN, dict(files=files, records=records, runtime=prior['runtime'], room=room,
                         threshold_requests=['DEFAULT'], models=MODELS, versions=versions, prime_samples=81600,
                         policy='All 12 consumed Mac sources, paired full-convolution and equal-length direct controls, equalized energy and shared anti-clipping gain. C161 and Binary40 fixed 235/256,5,67,half rearm and frozen dual arbiter. Official Willow DEFAULT native512, 32256 firmware prime plus49344 additional zero samples. Fresh instance per complete replay. Separate own and reciprocal events by native clock. Host synthetic development. No new independent exposure, acoustic hardware, tuning or optimizer updates.'))


def run():
    import importlib.metadata as metadata
    from features import features

    spec = read(PLAN)
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    for path, digest in spec['files'].items():
        assert sha(path) == digest, path
    assert all(metadata.version(k) == v for k, v in spec['versions'].items())
    evidence = BASE / 'evidence'
    evidence.mkdir(exist_ok=True)
    rows = []
    for number, record in enumerate(spec['records']):
        prefix = f'{number:02}'
        pcm, rate = sf.read(record['path'], dtype='int16')
        assert rate == 16000 and pcm.shape == (record['samples'],) and sha(record['path']) == record['sha256']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == record['pcm_sha256']
        output = evidence / prefix
        if not output.exists():
            tmp = Path(tempfile.mkdtemp(prefix='nyx-mac12-room-')) / 'case'
            args = [sys.executable, __file__, 'willow', '--model', 'willow', '--threshold', 'DEFAULT',
                    '--record-index', str(number), '--output', str(tmp)]
            for k, v in spec['runtime'].items():
                args.extend(['--' + k, v])
            with (tmp.parent / 'launcher.log').open('xb') as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert read(tmp/'report.json')['source_pcm_sha256'] == record['pcm_sha256']
            (tmp/'input.pcm').unlink()
            (tmp.parent/'launcher.log').rename(tmp/'launcher.log')
            shutil.move(str(tmp), str(output))
            tmp.parent.rmdir()
        w = read(output/'report.json')
        assert w['execution_sha256'] == sha(PLAN) and w['record_index'] == number and w['record'] == record
        assert w['model'] == 'wn9_heywillow_tts' and w['mode'] == 'DEFAULT' and w['stage'] == 104
        assert w['actual_float32_bits'] == w['post_prime_getter_bits'] == '3f24dd2f'
        assert w['source_pcm_sha256'] == record['pcm_sha256'] and w['total_prime_samples'] == 81600
        assert w['input_crc32'] == format(zlib.crc32(bytes(49344*2) + pcm.astype('<i2').tobytes()), '08x')
        raw_returns = (output/'raw-returns.bin').read_bytes()
        assert sha(output/'raw-returns.bin') == w['raw_returns_sha256'] and sha(output/'serial.log') == w['serial_sha256']
        assert len(raw_returns) == (len(pcm) + 49344 + 511)//512 and set(raw_returns) <= {0, 1, 2}
        native_events = [min(i*512, len(pcm)+49344)-49344 for i, value in enumerate(raw_returns, 1) if value == 2]
        assert [e['relative_sample'] for e in w['real_events']] == [v for v in native_events if v > 0]

        combined = np.r_[np.zeros(81600, np.int16), pcm]
        frontend = features(combined)
        values, support, _ = prepare(frontend, len(combined))
        clock = np.minimum(((support+479)//480)*480, len(combined))-81600
        codes, events, emitted, suppressed, timing = dual(values, clock)
        for name in MODELS:
            steps = event_steps(codes[name], 235/256)
            assert events[name] == [dict(step=s, relative_sample=int(clock[s-1])) for s in steps]
        trace = output/'nyx.npz'
        np.savez_compressed(trace, **codes, relative_block_return=clock)
        outcomes = {name: [e['relative_sample'] for e in ev if e['relative_sample'] > 0]
                    for name, ev in {**events, 'dual': emitted}.items()}
        outcomes['willow'] = [v for v in native_events if v > 0]
        rows.append(dict(index=record['index'], condition=record['condition'], voice=record['voice'],
                         form=record['form'], target=record['target'], pcm_sha256=record['pcm_sha256'],
                         outcomes=outcomes, events=events, emitted=emitted, suppressed=suppressed, timing=timing,
                         willow_firmware_prime_events=sum(v == 1 for _, v in w['prime_returns']),
                         willow_post_firmware_prime_events=[v for v in native_events if v <= 0],
                         willow_report_sha256=sha(output/'report.json'), nyx_trace_sha256=sha(trace)))
        print(prefix, record['condition'], record['target'], {k:len(v) for k,v in outcomes.items()}, flush=True)

    groups = {}
    for row in rows:
        for detector, times in row['outcomes'].items():
            own = row['target'] == ('hey_willow' if detector == 'willow' else 'hey_nyx')
            key = '/'.join([row['condition'], detector, 'own' if own else 'reciprocal'])
            group = groups.setdefault(key, dict(clips=0, hits=0))
            group['clips'] += 1
            group['hits'] += bool(times)
    members = {str(p.relative_to(evidence)):sha(p) for p in sorted(evidence.rglob('*')) if p.is_file()}
    with tarfile.open(ARTIFACT, 'x:gz') as tar:
        for name in members:
            tar.add(evidence/name, arcname=name)
    with tarfile.open(ARTIFACT) as tar:
        assert sorted(tar.getnames()) == sorted(members)
        for member in tar:
            assert hashlib.file_digest(tar.extractfile(member), 'sha256').hexdigest() == members[member.name]
    write_new(REPORT, dict(plan_sha256=sha(PLAN), records=rows, groups=groups, members=members,
                           archive_sha256=sha(ARTIFACT), archive_bytes=ARTIFACT.stat().st_size,
                           limits=spec['policy']))


if __name__ == '__main__':
    if sys.argv[1:2] == ['willow']:
        del sys.argv[1]
        official.PLAN = PLAN
        official.case()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('action', choices=['plan', 'run'])
        sys.path.insert(0, str((ROOT/'training').resolve()))
        {'plan': plan, 'run': run}[parser.parse_args().action]()
