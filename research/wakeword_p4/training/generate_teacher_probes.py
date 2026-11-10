"""Generate training-speaker-only audio for pretrained-model reconstruction checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from generate import CHECKPOINT_SHA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    source = a.upstream.resolve()
    checkpoint = source/'models/en_US-libritts_r-medium.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError('Changed Piper checkpoint')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for this synthesis probe')
    torch.set_num_threads(4); torch.manual_seed(10727)
    sys.path.insert(0, str(source))
    from generate_samples import generate_audio, get_phonemes, audio_float_to_int16
    model = torch.load(checkpoint, map_location='cpu', weights_only=False).eval().cuda()
    config_path = source/'models/en_US-libritts_r-medium.pt.json'
    config = json.loads(config_path.read_text())
    speakers = np.random.default_rng(726).permutation(904)[:4]
    a.output.mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for speaker in speakers:
            for text in ('Hey Jarvis.', 'Hey Jarvis, what time is it?', 'Hey Nicks.', 'Hey Next.'):
                phones = get_phonemes(config['espeak']['voice'], config, text)[0]
                audio, duration = generate_audio(model, torch.LongTensor([int(speaker)]),
                                                torch.LongTensor([int(speaker)]), [phones],
                                                .5, .667, .8, 1., None)
                wave = resample_poly(audio.cpu().numpy()[0, 0], 320, 441)
                end = min(len(wave), int(np.ceil(duration.cpu().numpy()[0, 0, :len(phones)].sum()*16000/22050)))
                pcm = audio_float_to_int16(wave[:end])
                if len(pcm) <= 1600 or np.abs(pcm.astype(np.int32)).max() <= 1000:
                    raise ValueError('Invalid synthesized waveform')
                name = f'{len(rows):03d}.wav'
                sf.write(a.output/name, pcm, 16000, subtype='PCM_16')
                rows.append(dict(path=name, speaker=int(speaker), text=text,
                                 teacher_target_text='Jarvis' in text, samples=len(pcm),
                                 sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest(),
                                 pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
    manifest = a.output/'manifest.json'
    manifest.write_text(json.dumps(rows, indent=2)+'\n')
    report = dict(role='Pretrained Hey Jarvis numerical probe audio only; not Nyx recognition data',
                  seed=10727, speaker_partition_seed=726, speakers=speakers.tolist(),
                  speakers_from_existing_piper_training_partition=True, clips=len(rows),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  checkpoint_sha256=CHECKPOINT_SHA,
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256((source/'generate_samples.py').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  validation_or_final_accessed=False,
                  limits='Expected text is synthesis intent, not an independent phonetic audit. Probe audio must not be mixed into Nyx recognition metrics.')
    (a.output/'generation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
