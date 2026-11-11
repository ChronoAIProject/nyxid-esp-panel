"""Bind the 40 reported PCM endpoint files to their original Mac manifest.

Read-only provenance/PCM accounting. No synthesis, listening, model scoring,
data edits or inference of native-waveform integrity from endpoint counts.
"""
import argparse
import array
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import wave


def sha(data):
    return hashlib.sha256(data).hexdigest()


def function_text(source, name):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError('Missing or ambiguous converter')
    return ast.get_source_segment(source, functions[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-dataset', 'pack', 'upstream', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-revision', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    revision = subprocess.check_output(['git', 'rev-parse', args.gpu_revision+'^{commit}'], text=True).strip()

    def gpu_file(name):
        data = subprocess.check_output(['git', 'show', revision+':research/wakeword_p4/results/'+name])
        return json.loads(data), sha(data)

    quality, quality_sha = gpu_file('4090-old-complete-pcm-quality-v1.json')
    mechanism, mechanism_sha = gpu_file('4090-old-piper-quantization-diagnostic-v1.json')
    failed = [row for row in quality['details'] if not row['checks']['no_saturated_pcm16']]
    assert len(failed) == 40 and all(row['engine'] == 'piper' for row in failed)
    sources_bytes = (args.pack/'sources.json').read_bytes()
    assert sha(sources_bytes) == quality['source_manifest_sha256']
    sources = {row['source_sha256']: row for row in json.loads(sources_bytes)}
    manifest_bytes = (args.original_dataset/'manifest.jsonl').read_bytes()
    manifest_rows = [json.loads(line) for line in manifest_bytes.decode().splitlines()]
    by_sha = {}
    for row in manifest_rows:
        by_sha.setdefault(row['sha256'], []).append(row)
    generation_bytes = (args.original_dataset/'generation.json').read_bytes()
    generation = json.loads(generation_bytes)
    checkpoint = args.upstream/'models/en_US-libritts_r-medium.pt'
    assert sha(checkpoint.read_bytes()) == generation['checkpoint_sha256']
    local_generator_bytes = (args.upstream/'generate_samples.py').read_bytes()
    upstream_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip()
    upstream_bytes = subprocess.check_output(['git', 'show', upstream_revision+':generate_samples.py'], cwd=args.upstream)
    local_converter = function_text(local_generator_bytes.decode(), 'audio_float_to_int16')
    original_converter = function_text(upstream_bytes.decode(), 'audio_float_to_int16')
    assert local_converter == original_converter == mechanism['converter']
    wrapper = Path(__file__).with_name('generate.py')
    wrapper_bytes = wrapper.read_bytes()
    wrapper_committed = subprocess.check_output([
        'git', 'show', 'ac3986ed9e013fc818093c6e180cd5605167a2e4:research/wakeword_p4/training/generate.py'])
    assert wrapper_bytes == wrapper_committed
    details = []
    for row in failed:
        matches = by_sha.get(row['source_sha256'], [])
        assert len(matches) == 1
        original = matches[0]
        relative = Path(original['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = args.original_dataset/relative
        data = path.read_bytes()
        assert sha(data) == row['source_sha256']
        with wave.open(str(path), 'rb') as handle:
            assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 16000)
            pcm_bytes = handle.readframes(handle.getnframes())
        pcm = array.array('h', pcm_bytes)
        if sys.byteorder != 'little':
            pcm.byteswap()
        canonical = sources[row['source_sha256']]
        assert len(pcm) == row['samples'] == original['samples']
        assert any(alias['pcm_sha256'] == sha(pcm_bytes) for alias in canonical['aliases'])
        assert original['split'] == 'training' and original['label'] == row['label']
        endpoints = [i for i, value in enumerate(pcm) if value in (-32768, 32767)]
        assert len(endpoints) == row['saturated_samples'] == 1
        at = endpoints[0]
        assert pcm[at] == 32767
        details.append(dict(source_sha256=row['source_sha256'], pcm_sha256=sha(pcm_bytes),
                            archived_path=row['path'], original_manifest_record=original,
                            samples=len(pcm), minimum=min(pcm), maximum=max(pcm),
                            endpoint_index=at, endpoint_value=pcm[at],
                            endpoint_neighbours=list(pcm[max(0, at-2):min(len(pcm), at+3)]),
                            eligible_v40_v41_task=row['label'] == 1 or row['phrase'] in
                            ('hey nick', 'hey next', 'hey mix')))
    # The generator writes PCM WAV/JSON only. Do not reconstruct missing float
    # samples from quantized WAV or substitute a new nondeterministic MPS run.
    native_suffixes = {'.npy', '.npz', '.pt', '.pth', '.f32', '.float'}
    native_files = [str(p.relative_to(args.original_dataset)) for p in args.original_dataset.rglob('*')
                    if p.is_file() and p.suffix.lower() in native_suffixes]
    result = dict(
        scope=__doc__, gpu_revision=revision, gpu_quality_report_sha256=quality_sha,
        gpu_mechanism_report_sha256=mechanism_sha,
        source_pack_manifest_sha256=sha(sources_bytes),
        original_manifest_sha256=sha(manifest_bytes), original_manifest_rows=len(manifest_rows),
        original_generation_sha256=sha(generation_bytes), original_generation=generation,
        upstream_revision=upstream_revision, upstream_source_sha256=sha(upstream_bytes),
        local_generator_sha256=sha(local_generator_bytes),
        converter_text=local_converter, converter_text_sha256=sha(local_converter.encode()),
        local_upstream_gpu_converter_text_exact=True,
        wrapper_source_sha256=sha(wrapper_bytes),
        wrapper_source_matches_commit='ac3986ed9e013fc818093c6e180cd5605167a2e4',
        original_checkpoint_sha256=sha(checkpoint.read_bytes()),
        original_manifest_wav_pcm_matches=len(details),
        eligible_task_sources=sum(row['eligible_v40_v41_task'] for row in details),
        native_float_files_found=native_files, native_float_integrity_verified=False,
        endpoint_check_failure_retracted=False, training_data_modified=False,
        script_sha256=sha(Path(__file__).read_bytes()), details=details,
        conclusion='All40 files bind to the original Mac synthetic manifest and original PCM. The identical peak-normalizing converter explains why an endpoint code can be normal; actual native-float integrity remains unverified.',
        limits=[
            'Original WAV/PCM provenance is verified. Original generation.json does not record an invocation/source hash, so this is not a cryptographically complete execution trace.',
            'No native float arrays were located in the original dataset. Original wrapper persists WAV/JSON, not prequantization audio; no new synthesis substitutes for the original waveforms.',
            'A single endpoint or absence of a plateau does not prove clipping or absence of distortion. Keep the previous mechanical-check failure unchanged.',
            'No model, score, training data, new/final source, speaker, microphone or board operation.'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('original_manifest_wav_pcm_matches',
          'eligible_task_sources', 'local_upstream_gpu_converter_text_exact',
          'native_float_files_found', 'native_float_integrity_verified', 'conclusion')}, indent=2))


if __name__ == '__main__':
    main()
