"""Hash-locked all48 training-only cross-family ASR; no label changes."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

R = Path('research/wakeword_p4/results')
ADMISSION = R/'4090-chatterbox-bare-training-admission-v1.json'
CENSUS = R/'4090-bare-label-support-census-v1.json'
PLAN = R/'4090-bare-cross-family-plan-v1.json'
MODEL = Path('.local/wakeword/wav2vec2-base-960h-v1')
OUT = Path('.local/wakeword/bare-cross-family-v1')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inputs():
    data = json.loads(ADMISSION.read_text())
    census = json.loads(CENSUS.read_text())
    assert census['source_sha256'] == sha(ADMISSION)
    rows = sorted([dict(r, original_status=s) for s in ('admitted', 'quarantined') for r in data[s]], key=lambda r:r['index'])
    assert [r['index'] for r in rows] == list(range(48))
    files = {str(p):sha(p) for p in (Path(__file__), ADMISSION, CENSUS, MODEL/'source.json')}
    source = json.loads((MODEL/'source.json').read_text())
    assert source['revision'] == '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
    assert source['files']['model.safetensors']['sha256'] == '8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a'
    for name, meta in source['files'].items():
        p = MODEL/name
        assert p.stat().st_size == meta['bytes'] and sha(p) == meta['sha256']
        files[str(p)] = meta['sha256']
    for r in rows:
        p = Path(r['wav']['path'])
        assert r['request']['role'] == 'training-only'
        assert p.stat().st_size == r['wav']['bytes'] and sha(p) == r['wav']['sha256'] == r['asr']['sha256']
        files[str(p)] = r['wav']['sha256']
    return rows, files


def main():
    rows, files = inputs()
    if sys.argv[1] == 'lock':
        with PLAN.open('x') as f:
            json.dump(dict(files=files, count=48, policy='All48 once CUDA FP32 greedy CTC; CPU2; no LM/VAD/prompt/resampling/retry, label changes, wake scoring or acceptance. Existing five aliases only.'), f, indent=2)
        return
    assert sys.argv[1] == 'run' and json.loads(PLAN.read_text())['files'] == files
    assert not OUT.exists()
    OUT.mkdir()
    os.environ['USE_TF'] = '0'
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    processor = Wav2Vec2Processor.from_pretrained(MODEL, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    assert next(model.parameters()).dtype == torch.float32
    aliases = {'hey nyx', 'hey nix', 'hey nicks', 'hey knicks', 'hey nick s'}
    with (OUT/'details.jsonl').open('x') as f:
        for r in rows:
            pcm, rate = sf.read(r['wav']['path'], dtype='float32')
            assert rate == 16000 and pcm.ndim == 1 and np.isfinite(pcm).all()
            values = processor(pcm, sampling_rate=rate, return_tensors='pt').input_values.cuda()
            with torch.inference_mode():
                text = processor.batch_decode(model(input_values=values).logits.argmax(-1).cpu())[0]
            normal = ' '.join(re.findall('[a-z]+', text.lower()))
            record = dict(index=r['index'], voice=r['request']['voice'], text=r['request']['text'], seed=r['request']['seed'], original_status=r['original_status'], wav_sha256=r['wav']['sha256'], transcript=text, ctc_agreement=normal in aliases, whisper_transcript=r['asr']['transcript'], whisper_agreement=r['asr']['text_agreement'])
            f.write(json.dumps(record)+'\n')
            f.flush()
            print(json.dumps(record), flush=True)
    report = dict(count=48, plan_sha256=sha(PLAN), details_sha256=sha(OUT/'details.jsonl'), torch=torch.__version__, gpu=torch.cuda.get_device_name(), dtype=str(next(model.parameters()).dtype), limits='Cross-family consistency only, not phone truth or label adjudication; original decisions unchanged.')
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
