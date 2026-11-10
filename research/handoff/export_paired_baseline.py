"""Export exact synthetic development PCM and its verified Hi ESP baseline.

Only the plan-bound paired scope is exported. Excluded acoustic controls are
not opened. Paths become relative; source WAV bytes and numeric results stay
unchanged. No microphone recording, account/device state or feature arrays.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import wave
import zlib

from export_assets import Archive, encoded, sha


FIXTURE_FIELDS = ('voice', 'target', 'text', 'variant', 'samples', 'keyword_end',
                  'sha256', 'scope')
RESULT_FIELDS = ('id', 'complete', 'samples', 'crc32', 'events', 'calls',
                 'total_us', 'max_us', 'offsets')


def verified_rows(manifest_path, baseline_path, plan):
    if (sha(manifest_path.read_bytes()) != plan['manifest_sha256']
            or sha(baseline_path.read_bytes()) != plan['baseline_sha256']):
        raise ValueError('Original manifest/baseline differs from the export plan')
    fixtures = list(map(json.loads, manifest_path.read_text().splitlines()))
    baseline = json.loads(baseline_path.read_text())
    if (not baseline['restored'] or baseline['sample_rate'] != 16000
            or baseline['manifest_sha256'] != plan['manifest_sha256']
            or len(fixtures) != len(baseline['clips'])
            or dict(Counter(r['scope'] for r in fixtures)) != plan['scope_counts']):
        raise ValueError('Incomplete baseline or unexpected source population')
    root = manifest_path.parent.resolve()
    rows, sources, names = [], [], set()
    for fixture, measured in zip(fixtures, baseline['clips']):
        if (any(fixture[k] != measured[k] for k in ('path', *FIXTURE_FIELDS))
                or not measured['result']['complete']):
            raise ValueError('Source/result identity or completion differs')
        if fixture['scope'] != plan['selected_scope']:
            continue
        source = Path(fixture['path'])
        source = (root / source).resolve() if not source.is_absolute() else source.resolve()
        relative = source.relative_to(root)
        if relative.as_posix() in names:
            raise ValueError('Duplicate paired source path')
        names.add(relative.as_posix())
        if sha(source.read_bytes()) != fixture['sha256']:
            raise ValueError('Synthetic WAV checksum differs')
        with wave.open(str(source), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                    wav.getcomptype(), wav.getnframes()) != (
                        1, 2, 16000, 'NONE', fixture['samples']):
                raise ValueError('Expected unchanged mono16kHz PCM16 WAV')
            pcm = wav.readframes(wav.getnframes())
        result = measured['result']
        if (result['samples'] != fixture['samples'] or zlib.crc32(pcm) != result['crc32']
                or result['events'] != len(result['offsets'])
                or result['calls'] != math.ceil(fixture['samples'] / 512)
                or any(type(at) is not int or not 0 <= at <= fixture['samples']
                       for at in result['offsets'])):
            raise ValueError('PCM transfer CRC, event offsets or inference coverage differs')
        row = {'path': relative.as_posix(), **{k: fixture[k] for k in FIXTURE_FIELDS}}
        rows.append({**row, 'threshold': measured['threshold'],
                     'result': {k: result[k] for k in RESULT_FIELDS}})
        sources.append(source)
    if (dict(Counter(r['target'] for r in rows)) != plan['target_counts']
            or sorted({r['voice'] for r in rows}) != plan['voices']
            or sorted({r['variant'] for r in rows}) != plan['variants']):
        raise ValueError('Selected paired population differs from the plan')
    return rows, sources, baseline


def export(manifest_path, baseline_path, plan_path, output, name):
    if Path(name).parts != (name,) or name in ('.', '..'):
        raise ValueError('Archive name must be one path component')
    plan = json.loads(plan_path.read_text())
    rows, sources, original = verified_rows(manifest_path, baseline_path, plan)
    manifest = ('\n'.join(json.dumps({k: v for k, v in r.items()
                                    if k not in ('threshold', 'result')}) for r in rows) + '\n').encode()
    baseline = {'scope': 'Paired synthetic development direct-PCM regression, not physical/final acceptance.',
                'sample_rate': 16000, 'engine': original['engine'], 'mode': original['mode'],
                'probe_sha256': original['probe_sha256'], 'manifest_sha256': sha(manifest),
                'original_manifest_sha256': plan['manifest_sha256'],
                'original_baseline_sha256': plan['baseline_sha256'],
                'excluded_scopes': {k: v for k, v in plan['scope_counts'].items()
                                    if k != plan['selected_scope']},
                'restored': True, 'clips': rows}
    provenance = {'scope': __doc__, 'plan_sha256': sha(plan_path.read_bytes()),
                  'exporter_sha256': sha(Path(__file__).read_bytes()),
                  'source_wavs': len(rows), 'target_counts': plan['target_counts'],
                  'voice_conditions': plan['voices'], 'variants': plan['variants'],
                  'original_manifest_sha256': plan['manifest_sha256'],
                  'original_baseline_sha256': plan['baseline_sha256'],
                  'portable_manifest_sha256': sha(manifest),
                  'all_source_wav_hashes_and_pcm_crcs_verified': True,
                  'audio_speed_changed': False, 'accepted': False,
                  'limits': plan['limits']}
    output.mkdir(parents=True, exist_ok=False)
    archive = Archive(output, name)
    prefix = 'workspace/hey-nyx-training-v3/paired-mac-regression-v1/'
    for row, source in zip(rows, sources):
        archive.add(prefix + row['path'], source=source)
    archive.add(prefix + 'manifest.jsonl', data=manifest)
    archive.add(prefix + 'baseline.json', data=encoded(baseline))
    archive.add(prefix + 'provenance.json', data=encoded(provenance))
    archive.add(prefix + 'export-plan.json', source=plan_path)
    entry = archive.close()
    (output / 'asset.json').write_bytes(encoded(entry))
    (output / 'report.json').write_bytes(encoded(provenance))
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', default='paired-mac-regression-v1')
    args = p.parse_args()
    print(json.dumps(export(args.manifest, args.baseline, args.plan, args.output, args.name)))


if __name__ == '__main__':
    main()
