"""Recover native Kokoro token durations and audit correspondence to dev PCM.

These are synthesis frame allocations, not manually verified phone boundaries.
The prospective mapping is cumulative duration*600 samples at24kHz, minus the
actual library trim offset. No wake scores or old endpoint labels are read.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import string

from kokoro_onnx import Kokoro
from kokoro_onnx.trim import trim
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf

from comparison_holdout import Session
from kokoro_samples import MODEL_SHA, VOICES_SHA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Capture(Session):
    def run(self, names, feed, run_options=None):
        outputs = super().run(names, feed, run_options)
        self.captured = (feed, outputs)
        return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generated', type=Path, required=True,
                   help='Fresh directory retaining actual diagnostic resynthesis WAVs')
    a = p.parse_args()
    if a.output.exists() or a.generated.exists():
        p.error('Fresh report and generated-waveform directory required')
    m = json.loads(a.manifest.read_text())
    if m.get('independent_final_audio_accessed') or not m.get('paired_wakephrases'):
        raise ValueError('Only original paired development inputs')
    for name, digest in [('kokoro-v1.0.onnx', MODEL_SHA), ('voices-v1.0.bin', VOICES_SHA)]:
        if sha(a.assets/name) != digest:
            raise ValueError('Changed synthesis assets')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = Capture(str(a.assets/'kokoro-v1.0.onnx'), options, providers=['CPUExecutionProvider'])
    if [o.name for o in session.get_outputs()] != ['waveform', 'duration']:
        raise ValueError('Unexpected native output schema')
    engine = Kokoro.from_session(session, str(a.assets/'voices-v1.0.bin'))
    a.generated.mkdir(parents=True)
    rows = []
    for row in m['records']:
        if row['split'] != 'validation' or row['role'] != 'development-paired-validation':
            raise ValueError('Nondevelopment input')
        if row['target'] not in ('hey_nyx', 'hi_esp'):
            continue
        path = (a.manifest.parent/row['path']).resolve()
        if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed or escaped source PCM')
        session.requested_speed = row['speed']
        language = 'en-gb' if row['voice'].startswith('b') else 'en-us'
        signal, rate = engine.create(row['text'], voice=row['voice'], speed=row['speed'], lang=language)
        feed, outputs = session.captured
        waveform, duration = outputs
        tokens = np.asarray(feed.get('input_ids', feed.get('tokens'))).reshape(-1)
        phonemes = engine.tokenizer.phonemize(row['text'], language)
        labels = ['<bos>'] + [c for c in phonemes if c in engine.tokenizer.vocab] + ['<eos>']
        expected_tokens = [0] + engine.tokenizer.tokenize(phonemes) + [0]
        if (rate != 24000 or len(duration) != len(tokens) or len(labels) != len(tokens)
                or tokens.tolist() != expected_tokens or np.any(duration < 1)
                or len(waveform) != int(duration.sum())*600):
            raise ValueError('Invalid native token/duration/sample mapping')
        trimmed, bounds = trim(waveform)
        np.testing.assert_array_equal(signal, trimmed)
        resampled = resample_poly(signal, 2, 3)
        pcm = np.clip(np.rint(np.pad(resampled, (28800, 16000))*32768), -32768, 32767).astype('<i2')
        old, old_rate = sf.read(path, dtype='int16')
        if old_rate != 16000 or hashlib.sha256(old.astype('<i2').tobytes()).hexdigest() != row['pcm_sha256']:
            raise ValueError('Changed original PCM')
        exact = np.array_equal(pcm, old)
        generated = a.generated/row['path']
        generated.parent.mkdir(parents=True, exist_ok=True)
        sf.write(generated, pcm, 16000, subtype='PCM_16')
        ends = np.cumsum(duration)*600
        starts = np.concatenate(([0], ends[:-1]))
        last = max(i for i, label in enumerate(labels[1:-1], 1)
                   if not label.isspace() and label not in string.punctuation)
        candidate = 1.8+(int(ends[last])-int(bounds[0]))/24000
        rows.append(dict(path=row['path'], target=row['target'], voice=row['voice'], speed=row['speed'],
                         wav_sha256=row['sha256'], pcm_sha256=row['pcm_sha256'], exact_original_pcm_reproduced=exact,
                         generated_wav_sha256=sha(generated),
                         generated_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                         original_samples=len(old), generated_samples=len(pcm),
                         matching_original_length=len(pcm) == len(old),
                         mismatched_samples=int(np.count_nonzero(pcm != old)) if len(pcm) == len(old) else None,
                         phonemes=phonemes, raw_samples_24k=len(waveform), trim_bounds_24k=[int(x) for x in bounds],
                         native_duration_sum_samples_matches=True,
                         duration_tokens=[dict(label=label, id=int(token), frames=int(d),
                                               raw_start_samples_24k=int(start), raw_end_samples_24k=int(end))
                                          for label, token, d, start, end in zip(labels, tokens, duration, starts, ends)],
                         last_nonpunctuation_token=last,
                         generated_candidate_synthesis_grid_end_seconds=candidate,
                         original_pcm_candidate_end_seconds=candidate if exact else None))
    if len(rows) != 24:
        raise ValueError('Expected24 existing development targets')
    result = dict(role='Native synthesis duration development audit, not phonetic truth or scoring',
                  source_sha256=sha(Path(__file__)), manifest_sha256=sha(a.manifest),
                  model_sha256=MODEL_SHA, voices_sha256=VOICES_SHA,
                  kokoro_onnx=importlib.metadata.version('kokoro-onnx'), onnxruntime=ort.__version__,
                  wrapper_sha256=sha(Path(__import__('kokoro_onnx').__file__)),
                  records=rows, files=len(rows), all_original_pcm_exactly_reproduced=all(r['exact_original_pcm_reproduced'] for r in rows),
                  nonmatching_original_pcm_files=sum(not r['exact_original_pcm_reproduced'] for r in rows),
                  old_windows_or_labels_changed=False, wake_scores_accessed=False,
                  independent_final_audio_accessed=False,
                  policy='Native duration per tokenizer symbol,600samples/frame at24kHz. Actual library trim offset subtracted; add fixed1.8s lead. Last nonspace/nonpunctuation symbol, no empirical/vocoder offset applied.',
                  limits='Synthesizer durations allocate latent frames, including stress/space symbols; not independent acoustic alignment or human phone truth. Vocoder timing offset and boundary spread remain unverified. Resynthesis may differ from original PCM; in that case no timing candidate is assigned to original audio. New waveform hashes and candidates are diagnostic only, not independent voices or score-selected labels. No old results relabeled, no recall/latency superiority or final evaluation.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(files=len(rows), nonmatching_original_pcm_files=result['nonmatching_original_pcm_files'])))


if __name__ == '__main__':
    main()
