"""Cross-family ASR diagnostic on frozen training WAVs, never a label filter."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s):
    return ' '.join(re.findall('[a-z]+', s.lower()))


def accepted(expected, label):
    prefixes = ['hey nyx', 'hey nix', 'hey nicks', 'hey knicks', 'hey nick s']
    if label:
        for prefix in prefixes:
            if expected == prefix or expected.startswith(prefix+' '):
                return {p+expected[len(prefix):] for p in prefixes}
    return {expected}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--whisper-details', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output directory required')
    manifest = a.dataset/'manifest.jsonl'
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    if len(rows) != 200 or any(r['split'] != 'training' for r in rows):
        raise ValueError('Require the frozen200 training-only pilot')
    if sha(manifest) != '78cf35c78f71c246a85eeabb285bedaafbe92a1980aaa21411741835f12e880b':
        raise ValueError('Historical pilot manifest changed')
    whisper = {r['path']: r for r in map(json.loads, a.whisper_details.read_text().splitlines())}
    if set(whisper) != {r['path'] for r in rows}:
        raise ValueError('Whisper comparison source set differs')
    source = json.loads((a.model/'source.json').read_text())
    for name, meta in source['files'].items():
        if sha(a.model/name) != meta['sha256']:
            raise ValueError('CTC model source hash mismatch')
    # Verify every WAV before inference, so no successful subset is reported.
    for r in rows:
        if sha(a.dataset/r['path']) != r['sha256'] or whisper[r['path']]['sha256'] != r['sha256']:
            raise ValueError('Audio changed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    processor = Wav2Vec2Processor.from_pretrained(a.model, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(a.model, local_files_only=True).cuda().eval()
    counts = Counter()
    a.output.mkdir(parents=True)
    with (a.output/'details.jsonl').open('w') as out:
        for i,r in enumerate(rows, 1):
            pcm, rate = sf.read(a.dataset/r['path'], dtype='float32')
            if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
                raise ValueError('Invalid original PCM')
            inputs = processor(pcm, sampling_rate=rate, return_tensors='pt')
            with torch.inference_mode():
                logits = model(input_values=inputs.input_values.cuda()).logits
            text = processor.batch_decode(logits.argmax(-1).cpu())[0]
            aliases = accepted(norm(r['text']), r['label'])
            match = norm(text) in aliases
            previous = norm(whisper[r['path']]['transcript']) in aliases
            group = 'positive' if r['label'] else 'negative'
            counts[group+'/clips'] += 1
            counts[group+'/ctc_alias_agreement'] += int(match)
            counts[group+'/whisper_alias_agreement'] += int(previous)
            counts[group+'/both_agree'] += int(match and previous)
            counts[group+'/both_disagree'] += int(not match and not previous)
            counts[group+'/cross_family_disagreement'] += int(match != previous)
            out.write(json.dumps(dict(path=r['path'], sha256=r['sha256'], expected=r['text'],
                label=r['label'], transcript=text, whisper_transcript=whisper[r['path']]['transcript'],
                ctc_agreement=match, whisper_agreement=previous))+'\n')
            out.flush()
            if i % 25 == 0:
                print(json.dumps({'completed': i, 'planned': len(rows)}), flush=True)
    report = dict(role='Frozen training-only cross-family ASR consistency diagnostic',
        manifest_sha256=sha(manifest), source_sha256=sha(Path(__file__)), ctc_source=source,
        whisper_details_sha256=sha(a.whisper_details), details_sha256=sha(a.output/'details.jsonl'),
        counts=dict(counts), runtime=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            model_dtype=str(next(model.parameters()).dtype), decoding='CTC greedy, no LM/prompt/VAD/resampling'),
        audio_or_labels_changed=False, wake_models_scored=False, final_data_used=False,
        limits='ASR agreement/disagreement is not pronunciation truth, phonetic endpoints or wake recall. Models have different vocabularies and training distributions; two recognizers agreeing can share errors. No filtering or automatic relabeling.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='ctc_source'}),flush=True)


if __name__ == '__main__':
    main()
