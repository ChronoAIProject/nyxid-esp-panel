"""Generate training-only near-phonetic negatives with pinned Piper voices."""
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
    torch.set_num_threads(4); torch.manual_seed(30726)
    sys.path.insert(0, str(source))
    from generate_samples import generate_audio, get_phonemes, audio_float_to_int16
    model = torch.load(checkpoint, map_location='cpu', weights_only=False).eval().cuda()
    config_path = source/'models/en_US-libritts_r-medium.pt.json'
    config = json.loads(config_path.read_text())
    speakers = np.random.default_rng(726).permutation(904)[:32]
    texts = [
        "Hey Nets.", "The nets are ready.", "They sat beside the nets.",
        "Please check the nets.", "We need new nets.", "He knits every day.",
        "Hey Neck.", "My neck feels stiff.", "Hey Necks.", "They mixed it well.",
        "Hey Mixes.", "She mixes the colors.", "Hey Next, please wait.",
        "The next one is ready.", "Hey Nice, good morning.", "Have a nice day.",
        "Hey Six, come here.", "There are six of them.", "Hey Nick, please stop.",
        "Nick said hello.", "Hey Lex.", "Hey Rex.", "Hey Nexus, are you ready?",
        "The network is connected.",
    ]
    a.output.mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for speaker in speakers:
            for text in texts:
                phones = get_phonemes(config['espeak']['voice'], config, text)[0]
                audio, duration = generate_audio(model, torch.LongTensor([int(speaker)]),
                                                torch.LongTensor([int(speaker)]), [phones],
                                                .5, .667, .8, 1., None)
                wave = resample_poly(audio.cpu().numpy()[0, 0], 320, 441)
                end = min(len(wave), int(np.ceil(duration.cpu().numpy()[0, 0, :len(phones)].sum()*16000/22050)))
                pcm = audio_float_to_int16(wave[:end])
                if len(pcm) <= 1600 or np.abs(pcm.astype(np.int32)).max() <= 1000:
                    raise ValueError('Invalid synthesized waveform')
                name = f'training/0/{len(rows):05d}.wav'
                (a.output/name).parent.mkdir(parents=True, exist_ok=True)
                sf.write(a.output/name, pcm, 16000, subtype='PCM_16')
                rows.append(dict(path=name, speakers=[int(speaker), int(speaker)], text=text,
                                 split='training', label=0, engine='piper', keyword_end=None,
                                 negative_family='near-phonetic-v20', samples=len(pcm),
                                 sha256=hashlib.sha256((a.output/name).read_bytes()).hexdigest(),
                                 pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
    manifest = a.output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = dict(role='Training-only synthetic near-phonetic negative addition; not evaluation evidence',
                  texts=texts, seed=30726, speaker_partition_seed=726, speakers=speakers.tolist(),
                  speakers_from_existing_piper_training_partition=True, clips=len(rows),
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  checkpoint_sha256=CHECKPOINT_SHA,
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256((source/'generate_samples.py').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  validation_or_final_accessed=False,
                  limits='Expected text is synthesis intent, not an independent phonetic audit. Only existing training-speaker identities are used. Do not score these as held-out recognition metrics.')
    (a.output/'generation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
