"""Mechanical positive control; LibriSpeech may overlap ASR training."""
import argparse
import json
from pathlib import Path
from audit_training_ctc import sha, norm


def distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = cur
    return prev[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    manifest = a.output/'manifest.jsonl'
    if a.prepare:
        a.output.mkdir(parents=True, exist_ok=False)
        rows = []
        for speaker in sorted(x for x in a.root.iterdir() if x.is_dir())[:24]:
            transcript = sorted(speaker.glob('*/*.trans.txt'))[0]
            line = sorted(transcript.read_text().splitlines())[0]
            key, expected = line.split(' ', 1)
            audio = transcript.parent/(key+'.flac')
            rows.append(dict(path=str(audio.relative_to(a.root)), expected=expected,
                             sha256=sha(audio), speaker=speaker.name))
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(json.dumps(dict(manifest_sha256=sha(manifest), clips=len(rows))))
        return
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    if (a.output/'report.json').exists():
        raise ValueError('Refuse overwrite')
    source = json.loads((a.model/'source.json').read_text())
    for name, meta in source['files'].items():
        assert sha(a.model/name) == meta['sha256']
    for r in rows:
        assert sha(a.root/r['path']) == r['sha256']
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    processor = Wav2Vec2Processor.from_pretrained(a.model, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(a.model, local_files_only=True).cuda().eval()
    totals = dict(word_errors=0, reference_words=0, character_errors=0, reference_characters=0)
    with (a.output/'details.jsonl').open('x') as out:
        for r in rows:
            pcm, rate = sf.read(a.root/r['path'], dtype='float32')
            assert rate == 16000 and pcm.ndim == 1 and np.isfinite(pcm).all()
            inputs = processor(pcm, sampling_rate=rate, return_tensors='pt')
            with torch.inference_mode():
                logits = model(input_values=inputs.input_values.cuda()).logits
            text = processor.batch_decode(logits.argmax(-1).cpu())[0]
            ref, hyp = norm(r['expected']), norm(text)
            totals['word_errors'] += distance(ref.split(), hyp.split())
            totals['reference_words'] += len(ref.split())
            totals['character_errors'] += distance(ref.replace(' ', ''), hyp.replace(' ', ''))
            totals['reference_characters'] += len(ref.replace(' ', ''))
            out.write(json.dumps(dict(**r, transcript=text, seconds=len(pcm)/rate))+'\n')
            out.flush()
    report = dict(clips=len(rows), speakers=len({r['speaker'] for r in rows}), **totals,
        wer=totals['word_errors']/totals['reference_words'],
        cer=totals['character_errors']/totals['reference_characters'],
        manifest_sha256=sha(manifest), details_sha256=sha(a.output/'details.jsonl'),
        script_sha256=sha(Path(__file__)), model_source=source,
        gpu=torch.cuda.get_device_name(), dtype=str(next(model.parameters()).dtype),
        decoding='FP32 greedy CTC, no LM/VAD/prompt/resampling; full PCM',
        normalization='lowercase ASCII letters; whitespace word boundaries; CER removes spaces',
        limits='Mechanical positive control only: train-clean-100 overlaps wav2vec2 LibriSpeech 960h training. Not held-out ASR performance, pronunciation truth, phonetic endpoints or wake-word acceptance.',
        final_data_used=False, wake_models_scored=False)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'model_source'}))


if __name__ == '__main__':
    main()
