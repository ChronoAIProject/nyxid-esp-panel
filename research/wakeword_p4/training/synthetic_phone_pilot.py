"""Training-only explicit-phone pilot, preserving full untrimmed native output."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from kokoro_onnx import Kokoro
from comparison_holdout import Session
from kokoro_samples import MODEL_SHA, VOICES_SHA


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Capture(Session):
    def run(self, names, feed, run_options=None):
        result = super().run(names, feed, run_options)
        self.calls.append((feed, result))
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-partitions', type=Path, help='Existing generator metadata; expand to every historical training voice only')
    p.add_argument('--development-partitions', type=Path, help='Generate only historical validation voices; never training/testing voices')
    a = p.parse_args()
    if a.training_partitions and a.development_partitions:
        p.error('Choose one scope')
    a.output.mkdir(parents=True, exist_ok=False)
    for name, expected in [('kokoro-v1.0.onnx', MODEL_SHA), ('voices-v1.0.bin', VOICES_SHA)]:
        assert sha(a.assets/name) == expected
    opt = ort.SessionOptions(); opt.intra_op_num_threads = 4; opt.inter_op_num_threads = 1
    session = Capture(str(a.assets/'kokoro-v1.0.onnx'), opt, providers=['CPUExecutionProvider'])
    assert [o.name for o in session.get_outputs()] == ['waveform', 'duration']
    engine = Kokoro.from_session(session, str(a.assets/'voices-v1.0.bin'))
    # Already-used TRAINING voices only: this pilot is never an independent test.
    voices = ['af_bella', 'am_onyx', 'bf_isabella', 'bm_fable']
    speeds = (.9, 1., 1.1)
    role = 'training-only-mechanical-pilot'
    partition_sha = None
    if a.training_partitions:
        partitions = json.loads(a.training_partitions.read_text())['voice_partitions']
        voices = partitions['training']
        assert len(voices) == len(set(voices)) == 20
        assert not set(voices) & (set(partitions['validation']) | set(partitions['testing']))
        partition_sha = sha(a.training_partitions)
        speeds = (.8, .9, 1., 1.1, 1.2)
        role = 'training-only-explicit-phone-expansion'
    split = 'training'
    if a.development_partitions:
        partitions = json.loads(a.development_partitions.read_text())['voice_partitions']
        voices = partitions['validation']
        assert len(voices) == len(set(voices)) == 4
        assert not set(voices) & (set(partitions['training']) | set(partitions['testing']))
        partition_sha = sha(a.development_partitions)
        speeds = (.85, .95, 1.05, 1.15)
        role = 'development-only-explicit-phone-paired'
        split = 'validation'
    phrases = [('hey_nyx', 'hˈeɪ nˈɪks'), ('hey_nick', 'hˈeɪ nˈɪk'),
               ('hey_next', 'hˈeɪ nˈɛkst'), ('hey_mix', 'hˈeɪ mˈɪks')]
    if a.development_partitions:
        phrases.append(('hi_esp', 'hˈaɪ ˈiː ˈɛs pˈiː'))
    rows = []
    for voice in voices:
        for speed in speeds:
            for target, phones in phrases:
                assert all(c in engine.tokenizer.vocab for c in phones)
                session.requested_speed = speed; session.calls = []
                native, rate = engine.create(phones, voice=voice, speed=speed, is_phonemes=True, trim=False)
                assert len(session.calls) == 1 and rate == 24000
                feed, outputs = session.calls[0]
                tokens = np.asarray(feed.get('input_ids', feed.get('tokens'))).reshape(-1)
                duration = np.asarray(outputs[1]).reshape(-1)
                assert tokens.tolist() == [0]+engine.tokenizer.tokenize(phones)+[0]
                assert len(tokens) == len(duration) and np.all(duration >= 1)
                assert len(native) == int(duration.sum())*600
                np.testing.assert_array_equal(native, np.asarray(outputs[0]))
                stem = f'{voice}-{speed:.2f}-{target}'
                raw = a.output/(stem+'.npy'); np.save(raw, native)
                pcm = resample_poly(native, 2, 3)
                wav = a.output/(stem+'.wav'); sf.write(wav, pcm, 16000, subtype='PCM_16')
                finite = bool(np.isfinite(native).all() and np.isfinite(pcm).all())
                peak = float(np.max(np.abs(pcm))); rms = float(np.sqrt(np.mean(pcm**2)))
                clipped = int(np.count_nonzero(np.abs(pcm) >= 1))
                # Preregistered mechanical gates, not listening or phoneme truth.
                passed = finite and .2 <= len(pcm)/16000 <= 5 and rms >= .001 and clipped == 0
                rows.append(dict(role=role, split=split, voice=voice, speed=speed,
                    target=target, phonemes=phones, token_ids=tokens.tolist(), native_durations=duration.tolist(),
                    native_samples=len(native), samples=len(pcm), native_rate=rate, rate=16000,
                    raw_path=raw.name, raw_sha256=sha(raw), path=wav.name, sha256=sha(wav),
                    peak=peak, rms=rms, clipped_samples=clipped, quality_pass=passed,
                    generated_reference_end_sample=len(pcm), trimmed=False))
        print(json.dumps({'voice':voice,'completed':len(rows)}), flush=True)
    manifest = a.output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = dict(clips=len(rows), passed=sum(r['quality_pass'] for r in rows), voices=voices,
        model_sha256=MODEL_SHA, voices_sha256=VOICES_SHA, script_sha256=sha(Path(__file__)),
        role=role, speeds=speeds, historical_partitions_sha256=partition_sha,
        kokoro_version=importlib.metadata.version('kokoro-onnx'), manifest_sha256=sha(manifest),
        gates='finite; mono16k; duration[0.2,5]s; RMS>=0.001; zero abs(sample)>=1; complete native output; exact input tokens and duration mapping',
        limits='Historical split voice IDs reused, same generator; not new source-independent or untouched-voice data. Generated end is whole untrimmed output end, not final phoneme truth. Explicit phones constrain input, not acoustic correctness.',
        wake_models_scored=False, final_data_used=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
