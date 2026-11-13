"""Compare new full-recording hashes with actual training ambient and human ledgers."""
import hashlib
import json
from pathlib import Path
import soundfile as sf


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    base = Path('.local/wakeword/features-gpu-ami-extra-v1')
    target = Path('.local/wakeword/prospective-background-whole-waveform-v1/manifest.json')
    targets = json.loads(target.read_text())['records']
    new_encoded = {r['sha256'] for r in targets}
    new_pcm = {r['pcm_sha256'] for r in targets}
    training_metadata = json.loads(Path('.local/wakeword/model-per-draw-noise-fixed-formal-v1/training.json').read_text())['feature_metadata_sha256']
    for name in ['human.json', 'ambient.json']:
        path = base / name
        assert sha(path) == training_metadata[name]
    human = [r for r in json.loads((base / 'human.json').read_text()) if r['split'] == 'training']
    ambient = [r for r in json.loads((base / 'ambient.json').read_text()) if r['split'] == 'training']
    original = Path('.local/wakeword/backgrounds')
    background_manifest = original / 'manifest.json'
    lookup = {r['sha256']: original / r['path']
              for r in json.loads(background_manifest.read_text())['records'] if r['split'] == 'training'}
    ambient_details = []
    for row in ambient:
        path = base / row['audio_path'] if 'audio_path' in row else lookup[row['source_sha256']]
        assert sha(path) == row['source_sha256']
        pcm, rate = sf.read(path, dtype='int16')
        info = sf.info(path)
        eligible = pcm.ndim == 1 and rate == 16000 and info.subtype == 'PCM_16'
        digest = hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
        ambient_details.append(dict(path=str(path), encoded_sha256=row['source_sha256'],
            decoded_pcm_sha256=digest, comparable_mono_pcm16_16k=eligible,
            encoded_overlap=row['source_sha256'] in new_encoded,
            pcm_overlap=eligible and digest in new_pcm))
    streams = {r['stream_pcm_sha256'] for r in human if 'stream_pcm_sha256' in r}
    source_pcm = {s['pcm_sha256'] for r in human for s in r.get('sources', []) if 'pcm_sha256' in s}
    source_encoded = {s['sha256'] for r in human for s in r.get('sources', []) if 'sha256' in s}
    missing = [dict(path=r['path'], speaker=r['speaker'], files=r['files'])
               for r in human if 'stream_pcm_sha256' not in r]
    report = dict(script_sha256=sha(Path(__file__)), new_manifest_sha256=sha(target),
        current_training_index_sha256={name: sha(base/name) for name in ['human.json', 'ambient.json']},
        source_background_manifest_sha256=sha(background_manifest), new_recordings=len(targets),
        training_ambient_sources_decoded=len(ambient_details),
        all_ambient_comparable=all(r['comparable_mono_pcm16_16k'] for r in ambient_details),
        ambient_encoded_overlaps=sum(r['encoded_overlap'] for r in ambient_details),
        ambient_pcm_overlaps=sum(r['pcm_overlap'] for r in ambient_details),
        training_human_rows=len(human), human_stream_pcm_hashes=len(streams),
        human_source_pcm_hashes=len(source_pcm), human_source_encoded_hashes=len(source_encoded),
        human_stream_overlaps=sorted(streams & new_pcm),
        human_source_pcm_overlaps=sorted(source_pcm & new_pcm),
        human_source_encoded_overlaps=sorted(source_encoded & new_encoded),
        human_rows_without_stream_pcm_hash=missing, ambient_details=ambient_details,
        all_training_pcm_isolation_proven=False, background_qualified=False,
        detector_scored=False, final_audio_accessed=False,
        limits='Full-container byte/PCM equality only against actual training ambient originals and available '
        'human source/stream ledgers. Missing legacy human hashes remain a gap; synthetic phrase packs '
        'are not covered here. Re-encoding/cropping/partial overlap and physical people are not ruled out. '
        'No source-qualified development or scoring authorization.')
    output = Path('research/wakeword_p4/results/4090-new-background-training-overlap-v1.json')
    with output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['ambient_details', 'human_rows_without_stream_pcm_hash']}), flush=True)


if __name__ == '__main__': main()
