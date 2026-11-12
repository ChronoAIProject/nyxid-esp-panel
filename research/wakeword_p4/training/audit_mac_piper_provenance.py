"""Trace old Piper training metadata without audio decoding or synthesis.

Current source/configuration observations do not prove historical execution.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('generation-root', 'upstream', 'pack', 'tts-lock', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu-revision', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required')
    pinned = {
        a.generation_root/'manifest.jsonl': '914e0122768048976927a744c39b2ff5791e9aaa80392c2b1d1096f4edee1402',
        a.generation_root/'generation.json': '0db2ce585aa159a572494ec68806ded2125460fadb8741264daa0028f58c0d98',
        a.pack/'sources.json': '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981',
        a.upstream/'models/en_US-libritts_r-medium.pt': 'e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6',
        a.upstream/'models/en_US-libritts_r-medium.pt.json': '119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848',
    }
    for path, expected in pinned.items():
        if sha(path) != expected:
            raise ValueError('Metadata/model identity changed: '+path.name)
    def gpu_json(name):
        raw = subprocess.check_output(['git', 'show',
            f'{a.gpu_revision}:research/wakeword_p4/results/{name}'])
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    quality, quality_sha = gpu_json('4090-old-complete-pcm-quality-v1.json')
    mechanism, mechanism_sha = gpu_json('4090-old-piper-quantization-diagnostic-v1.json')
    sources = json.loads((a.pack/'sources.json').read_bytes())
    assert quality['source_manifest_sha256'] == sha(a.pack/'sources.json')
    piper = [s for s in sources if s['aliases'][0]['engine'] == 'piper']
    index = defaultdict(list)
    for line in (a.generation_root/'manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['split'] == 'training':
            index[row['sha256']].append(row)
    failures = [r for r in quality['details'] if not r['checks']['no_saturated_pcm16']]
    assert len(piper) == 2039 and len(failures) == 40
    assert all(s['source_sha256'] in index for s in piper)
    assert all(s['source_sha256'] in index for s in failures)
    generation = json.loads((a.generation_root/'generation.json').read_bytes())
    assert generation['checkpoint_sha256'] == sha(a.upstream/'models/en_US-libritts_r-medium.pt')
    upstream_source = a.upstream/'generate_samples.py'
    text = upstream_source.read_text()
    function = next(n for n in ast.parse(text).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'audio_float_to_int16')
    converter = ast.get_source_segment(text, function)
    assert converter == mechanism['converter']
    generator = Path(__file__).with_name('generate.py')
    assert sha(generator) == '747e7b0102c159776ed721b4d8cb60f4f02d7aa3e4d875bdbd53c5c90c88053f'
    extra_files = sorted(str(path.relative_to(a.generation_root))
                         for path in a.generation_root.rglob('*')
                         if path.is_file() and path.suffix.lower() != '.wav'
                         and path.name not in ('manifest.jsonl', 'generation.json'))
    lock = a.tts_lock.read_text()
    versions = dict(re.findall(r'^(numpy|piper-phonemize-cross|torch|torchaudio)==([^\s]+)$',
                               lock, re.MULTILINE))
    report = dict(
        scope='Mac original Piper training metadata provenance receipt; no historical native-float quality proof.',
        source_sha256=sha(Path(__file__)), gpu_revision=a.gpu_revision,
        gpu_quality_report_sha256=quality_sha, gpu_mechanism_report_sha256=mechanism_sha,
        original_manifest_sha256=sha(a.generation_root/'manifest.jsonl'),
        original_generation_sha256=sha(a.generation_root/'generation.json'),
        original_generation={k: generation[k] for k in ('phrase', 'sample_rate', 'device', 'checkpoint_sha256')},
        canonical_piper_sources=2039, matched_to_original_training_manifest=2039,
        endpoint_failures=40, endpoint_failures_matched=40,
        endpoint_failure_training_metadata=[dict(source_sha256=r['source_sha256'],
            original_training_rows=index[r['source_sha256']]) for r in failures],
        current_generator_sha256=sha(generator), current_upstream_source_sha256=sha(upstream_source),
        current_converter_exactly_matches_gpu_report=True,
        current_converter=converter,
        checkpoint_sha256=generation['checkpoint_sha256'],
        current_checkpoint_config_sha256=sha(a.upstream/'models/en_US-libritts_r-medium.pt.json'),
        current_tts_lock_sha256=sha(a.tts_lock), current_tts_lock_versions=versions,
        other_files_in_original_dataset_directory=extra_files,
        native_float_status='Not found in original dataset directory; reviewed generator persists PCM WAV/JSON only. No claim of global deletion.',
        original_per_file_fields=sorted({k for rows in index.values() for r in rows for k in r}),
        limits=[
            'Manifest SHA matches establish original training metadata provenance, not a fresh rehash/decode of those WAVs.',
            'Historical generation records omit runtime package/source/config hashes and native-float waveforms. Current lock/source/config are observations, not proof they executed for the original batch.',
            'Matching current converter text supports the normalization mechanism only; it cannot recover original float arrays or prove they were distortion-free.',
            'No inference, audio decoding/playback/recording, regeneration, filtering, attenuation, threshold changes, board or production operations.',
            'Only training rows are inspected/exported. Other splits have no waveform/feature/score access.',
            'Original40 endpoint failures remain unchanged. This receipt does not relax a prior gate or advance a model.',
        ])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('canonical_piper_sources',
        'matched_to_original_training_manifest', 'endpoint_failures_matched', 'native_float_status')}))


if __name__ == '__main__':
    main()
