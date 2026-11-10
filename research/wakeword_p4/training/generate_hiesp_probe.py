"""Fresh developer PCM for Hi ESP transport diagnostics; never final evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf
from kokoro_onnx import Kokoro
from kokoro_samples import MODEL_SHA, VOICES_SHA
from comparison_holdout import Session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--voice-partitions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    for name, expected in [('kokoro-v1.0.onnx', MODEL_SHA), ('voices-v1.0.bin', VOICES_SHA)]:
        if hashlib.sha256((a.assets/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Asset hash mismatch')
    voices = ['af_heart', 'am_michael']
    partitions = json.loads(a.voice_partitions.read_text())['voice_partitions']
    if not set(voices).issubset(partitions['training']):
        raise ValueError('Use existing training voices only')
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
    session = Session(str(a.assets/'kokoro-v1.0.onnx'), opts, providers=['CPUExecutionProvider'])
    model = Kokoro.from_session(session, str(a.assets/'voices-v1.0.bin'))
    a.output.mkdir(parents=True)
    rows = []
    for voice in voices:
        for index, text in enumerate(['Hi E S P.', 'Hey Nicks.', 'The weather is pleasant today.']):
            signal, rate = model.create(text, voice=voice, speed=1., lang='en-us')
            divisor = math.gcd(rate, 16000)
            signal = resample_poly(signal, 16000//divisor, rate//divisor)
            padded = np.pad(signal, (28800, 16000))
            path = a.output/f'{voice}-{index}.wav'
            sf.write(path, padded, 16000, subtype='PCM_16')
            pcm, rate = sf.read(path, dtype='int16')
            rows.append(dict(path=path.name, text=text, voice=voice, role='development-transport-probe',
                             synthesis_target=index == 0, samples=len(pcm), lead_samples=28800,
                             synthesis_end_samples=28800+len(signal),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()))
    report = dict(role='Synthetic training-voice developer transport probes, not acceptance',
                  model_sha256=MODEL_SHA, voices_sha256=VOICES_SHA,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  voice_partition_sha256=hashlib.sha256(a.voice_partitions.read_bytes()).hexdigest(),
                  independent_final_audio_accessed=False, records=rows,
                  limits='Expected text is synthesis intent, not audited pronunciation or phonetic alignment. Not recall/FAR evidence.')
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
