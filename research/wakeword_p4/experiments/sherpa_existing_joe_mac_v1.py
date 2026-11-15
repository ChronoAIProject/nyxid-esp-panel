"""Native keyword behavior on the existing, consumed Joe development waveforms.

No synthesis, speaker playback, microphone recording, training or board access.
The model and decoder defaults are the original GPU Jenny reference's settings;
this is a different speech source, not a cross-platform numerical comparison.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import sherpa_onnx
import soundfile as sf


MODELS = {
    'encoder': 'encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
    'decoder': 'decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    'joiner': 'joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx',
}
PARAMETERS = dict(provider='cpu', num_threads=2, sample_rate=16000,
                  feature_dim=80, max_active_paths=4, keywords_score=1.0,
                  keywords_threshold=0.25, num_trailing_blanks=1)
TOKENS = ['\u2581HE', 'Y', '\u2581', 'N', 'IC', 'K', 'S']
PRIME, CHUNK, FINISH = 81600, 480, 10560


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['records']
    if len(rows) != 360 or Counter(row['target'] for row in rows) != {
            'hey_nyx': 72, 'hi_esp': 72, 'confusion': 216}:
        raise ValueError('Expected the complete existing Joe360 cohort')
    if any(row['voice'] != 'joe' or row['split'] != 'validation' for row in rows):
        raise ValueError('Expected consumed Joe development inputs only')
    args.output.mkdir(exist_ok=False)
    keywords = args.output / 'keywords.txt'
    keywords.write_text(' '.join(TOKENS) + ' @HEY_NICKS\n')
    models = {name: str(args.model_dir / filename) for name, filename in MODELS.items()}
    config = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(args.manifest), source_sha256=sha(__file__),
        model_sha256={name: sha(path) for name, path in models.items()},
        tokens_sha256=sha(args.model_dir / 'tokens.txt'), keyword='HEY NICKS',
        keyword_tokens=TOKENS, parameters=PARAMETERS,
        protocol=dict(prime_samples=PRIME, chunk_samples=CHUNK, finish_samples=FINISH,
                      reset_at_input_transition=False, native_reset_after_every_keyword=True),
        versions={name: importlib.metadata.version(name) for name in
                  ['sherpa-onnx', 'sherpa-onnx-core', 'numpy', 'soundfile']},
        python=platform.python_version(), machine=platform.machine(),
        policy='One native CPU reference run of existing consumed Joe360 full PCM. '
               'No crop, resampling, new audio, TTS, training, final data, threshold/alias '
               'search, numerical agreement gate or P4 deployment. Generator text is '
               'intended speech, not verified phonetic truth. Existing compact-model '
               'Joe misses motivated this reference; it is not independent acceptance.',
    )
    (args.output / 'configuration.json').write_text(json.dumps(config, indent=2) + '\n')
    kws = sherpa_onnx.KeywordSpotter(tokens=str(args.model_dir / 'tokens.txt'),
                                   keywords_file=str(keywords), **models, **PARAMETERS)
    records = []
    started = time.monotonic()
    with (args.output / 'events.jsonl').open('x') as log:
        for index, row in enumerate(rows):
            pcm, rate = sf.read(args.manifest.parent / row['path'], dtype='int16')
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']:
                raise ValueError('Unexpected waveform format or length')
            if hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']:
                raise ValueError('Existing waveform identity changed')
            stream = kws.create_stream()
            events, fed = [], 0
            for segment in [np.zeros(PRIME, np.int16), pcm, np.zeros(FINISH, np.int16)]:
                for offset in range(0, len(segment), CHUNK):
                    wave = segment[offset:offset + CHUNK].astype(np.float32) / 32768
                    stream.accept_waveform(rate, wave)
                    fed += len(wave)
                    while kws.is_ready(stream):
                        kws.decode_stream(stream)
                        keyword = kws.get_result(stream)
                        if keyword:
                            relative = fed - PRIME
                            events.append(dict(keyword=keyword, return_relative_sample=relative,
                                               prime=relative <= 0, finish=relative > len(pcm)))
                            kws.reset_stream(stream)
            stream.input_finished()
            while kws.is_ready(stream):
                kws.decode_stream(stream)
                keyword = kws.get_result(stream)
                if keyword:
                    events.append(dict(keyword=keyword, return_relative_sample=fed - PRIME,
                                       prime=False, finish=True))
                    kws.reset_stream(stream)
            record = dict(index=index, record={key: row[key] for key in
                          ['id', 'target', 'text', 'length_scale', 'draw', 'variant',
                           'pcm_sha256', 'samples']}, events=events)
            records.append(record)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if (index + 1) % 18 == 0 or index == 0:
                print(json.dumps(dict(completed_inputs=index + 1, total_inputs=len(rows),
                                      elapsed_s=round(time.monotonic() - started, 2))), flush=True)
    groups = {}
    for group in ['hey_nyx', 'hi_esp', 'confusion']:
        selected = [record for record in records if record['record']['target'] == group]
        groups[group] = dict(
            inputs=len(selected),
            detected_inputs=sum(any(not event['prime'] for event in record['events'])
                                for record in selected),
            before_finish_inputs=sum(any(not event['prime'] and not event['finish']
                                         for event in record['events']) for record in selected),
            events=sum(not event['prime'] for record in selected for event in record['events']),
        )
    per_text = {}
    for text in sorted({record['record']['text'] for record in records}):
        selected = [record for record in records if record['record']['text'] == text]
        per_text[text] = dict(inputs=len(selected), detected_inputs=sum(
            any(not event['prime'] for event in record['events']) for record in selected))
    report = dict(schema='mac-sherpa-existing-joe-reference-v1', configuration=config,
                  completed=True, elapsed_s=round(time.monotonic() - started, 2),
                  groups=groups, per_text=per_text, records=records,
                  prime_events=sum(event['prime'] for record in records for event in record['events']),
                  finish_events=sum(event['finish'] for record in records for event in record['events']),
                  limits='Native ONNX reference only, not a P4 candidate or acoustic test. '
                         'One consumed source cannot validate pronunciation or generalization. '
                         'No noise-background FAR exposure was added. Native event/reset/finish '
                         'policy differs from the compact detector. No independent acceptance.',
                  candidate_promoted=False, full_goal_complete=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(completed=True, groups=groups, elapsed_s=report['elapsed_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
