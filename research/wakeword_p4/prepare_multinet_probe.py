"""Prepare a frozen training-source MultiNet feasibility probe without playback.

Original PCM speed, labels and source partitions stay intact. This diagnostic
does not score held-out audio or establish whole-application wake acceptance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
import wave
import zlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed_model(source, expected):
    files = sorted(p for p in source.iterdir() if p.is_file())
    if {p.name: sha(p) for p in files} != expected:
        raise ValueError('Pinned MultiNet model files differ')
    offset = 40 + 40 * len(files)
    header = struct.pack('<I32sI', 1, b'mn7_en', len(files))
    payload = b''
    for path in files:
        data = path.read_bytes()
        header += struct.pack('<32sII', path.name.encode('ascii'), offset, len(data))
        payload += data
        offset += len(data)
    return header + payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--parent-plan', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.output.exists() or sha(args.parent_plan) != plan['parent_plan_sha256']:
        p.error('Frozen parent plan and fresh private output required')
    if sha(Path(__file__)) != plan['preparation_script_sha256']:
        raise ValueError('Preparation script differs from frozen plan')
    template = Path(__file__).with_name('multinet_probe')
    for name, expected in plan['template_files'].items():
        if sha(template / name) != expected:
            raise ValueError('Firmware template differs from frozen plan')
    components = args.model.resolve().parents[3]
    dependency_manifest = 'dependencies:\n  idf: "==5.5.4"\n'
    for name, identity in plan['dependencies'].items():
        component = components / ('espressif__' + name)
        if (sha(component / 'idf_component.yml') != identity['manifest_sha256']
                or (component / '.component_hash').read_text().strip() != identity['component_hash']):
            raise ValueError('Pinned local dependency identity differs')
        dependency_manifest += (f'  espressif/{name}:\n    version: "=={identity["version"]}"\n'
                                f'    override_path: {json.dumps(str(component))}\n')
    parent = json.loads(args.parent_plan.read_text())
    groups = defaultdict(list)
    for source in parent['sources']:
        groups[(source['engine'], source['label'])].append(source)
    selected = [source for key in sorted(groups)
        for source in sorted(groups[key], key=lambda r: r['source_sha256'])[:plan['sources_per_engine_label']]]
    if selected != plan['sources']:
        raise ValueError('Source selection differs from frozen score-independent rule')
    base = args.data_root.resolve()
    jobs_path = base / 'head-features-tts-v1/jobs-private.json'
    if sha(jobs_path) != parent['source_jobs_sha256']:
        raise ValueError('Original source jobs differ')
    jobs = {r['id']: r for r in json.loads(jobs_path.read_text())}
    metadata_path = base / parent['source_text_metadata']
    if sha(metadata_path) != parent['source_text_metadata_sha256']:
        raise ValueError('Original source metadata differs')
    metadata = defaultdict(list)
    for row in json.loads(metadata_path.read_text()):
        if row['variant'] == 0:
            metadata[(metadata_path.parent / 'dataset' / row['path']).resolve()].append(row)
    heldout = set()
    for directory, expected in parent['source_job_files_sha256'].items():
        path = base / directory / 'jobs-private.json'
        if sha(path) != expected:
            raise ValueError('Feature source jobs differ')
        heldout.update(row['source_sha256'] for row in json.loads(path.read_text()) if row['split'] != 'training')
    if heldout.intersection(row['source_sha256'] for row in selected):
        raise ValueError('Selected training source overlaps held-out audio')
    pcm = struct.pack('<I', len(selected))
    records = []
    for number, source in enumerate(selected):
        job = jobs[source['variant0_id']]
        if (job['split'], job['label'], job['engine'], job['source_sha256'], job['variant']) != (
                'training', source['label'], source['engine'], source['source_sha256'], 0):
            raise ValueError('Selected source role differs')
        path = Path(job['source']).resolve()
        if sha(path) != source['source_sha256']:
            raise ValueError('Original WAV differs')
        aliases = metadata[path]
        if not aliases or any((r['split'], r['label'], r['engine'], r['sha256']) != (
                'training', source['label'], source['engine'], source['source_sha256']) for r in aliases):
            raise ValueError('Original text metadata role differs')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Expected original mono16kHz PCM16 source')
            signal = wav.readframes(wav.getnframes())
        rendered = bytes(plan['lead_samples'] * 2) + signal + bytes(plan['tail_samples'] * 2)
        pcm += struct.pack('<I', len(rendered) // 2) + rendered
        records.append({**source, 'clip': number, 'texts': sorted({r['text'] for r in aliases}),
            'samples': len(rendered) // 2, 'source_samples': len(signal) // 2,
            'keyword_end_sample': plan['lead_samples'] + job['keyword_end'] if source['label'] else None,
            'rendered_pcm_sha256': hashlib.sha256(rendered).hexdigest()})
    model = packed_model(args.model, plan['model_files'])
    args.output.mkdir(parents=True)
    shutil.copytree(template, args.output, dirs_exist_ok=True)
    (args.output / 'main/idf_component.yml').write_text(dependency_manifest)
    for name, raw in (('model', model), ('fixtures', pcm)):
        (args.output / 'main' / (name + '.bin')).write_bytes(struct.pack('<I', len(raw)) + zlib.compress(raw, 9))
    arms = plan['command_arms']
    header = '#pragma once\n'
    header += f'constexpr unsigned fixture_count = {len(records)};\n'
    header += f'constexpr unsigned detector_timeout_ms = {plan["detector_timeout_ms"]};\n'
    header += f'constexpr float detector_threshold = {plan["detector_threshold"]}f;\n'
    header += f'constexpr unsigned command_arm_count = {len(arms)};\n'
    header += 'constexpr unsigned command_counts[] = {' + ','.join(str(len(a['commands'])) for a in arms) + '};\n'
    header += 'constexpr const char *command_strings[][5] = {\n'
    for arm in arms:
        header += '{' + ','.join(json.dumps(command) for command in arm['commands']) + '},\n'
    header += '};\n'
    (args.output / 'main/fixture_config.h').write_text(header)
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'cases': len(records) * len(arms),
        'sources': records, 'command_arms': arms, 'detector_threshold': plan['detector_threshold'],
        'detector_timeout_ms': plan['detector_timeout_ms'], 'all_sources_training_only': True,
        'heldout_source_hash_disjoint': True, 'audio_speed_changed': False,
        'model_raw_bytes': len(model), 'fixtures_raw_bytes': len(pcm),
        'files': {name: sha(args.output / name) for name in ('main/model.bin', 'main/fixtures.bin', 'main/fixture_config.h')},
        'model_files': plan['model_files'], 'accepted': False}
    report['dependencies'] = plan['dependencies']
    (args.output / 'vectors-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
