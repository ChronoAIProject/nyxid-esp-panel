"""Prepare official public TIMIT sample as a non-wake alignment quality control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import soundfile as sf

EXPECTED = {
    'LDC93S1.wav': 'c7e8b51db10ec6a2aea8c92376f2b88af14c2b4950ed82db3d605a4b718c8f35',
    'LDC93S1.phn': '069015f10f74af9921c91cf1f203c8e4cdb77a259b12177fb8c3e43a9de3be5e',
    'LDC93S1.txt': '0b4ffafce18d6ec0df273c0189202a7c186cd0bbb9b40659490f5f2ad99e70a5',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    for name, digest in EXPECTED.items():
        if sha(a.source/name) != digest:
            raise ValueError('Changed official sample: ' + name)
    downloads = json.loads((a.source/'download.json').read_text())
    for row in downloads:
        if sha(a.source/row['path']) != row['sha256'] or not row['url'].startswith('https://catalog.ldc.upenn.edu/'):
            raise ValueError('Changed or nonofficial download provenance')
    pcm, rate = sf.read(a.source/'LDC93S1.wav', dtype='int16')
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Invalid reference PCM')
    start, end, text = (a.source/'LDC93S1.txt').read_text().split(maxsplit=2)
    if int(start) != 0 or int(end) != len(pcm):
        raise ValueError('Published text sample range does not match WAV')
    phones = []
    for line in (a.source/'LDC93S1.phn').read_text().splitlines():
        begin, finish, phone = line.split()
        begin, finish = int(begin), int(finish)
        if begin != (phones[-1]['end_samples'] if phones else 0) or not begin < finish <= len(pcm):
            raise ValueError('Invalid published phone intervals')
        phones.append(dict(label=phone, start_samples=begin, end_samples=finish))
    nonsilence = [r for r in phones if r['label'] not in ('h#', 'pau', 'epi')]
    if not nonsilence:
        raise ValueError('No published reference speech')
    transcript = text.strip().removesuffix('.').upper()
    a.output.mkdir(parents=True)
    relative = Path('ldc_sample')/'LDC93S1.wav'
    target = a.output/'corpus'/relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(a.source/'LDC93S1.wav', target)
    target.with_suffix('.lab').write_text(transcript+'\n')
    row = dict(path=str(relative), source_path='LDC93S1.wav', source_kind='published-human-phonetic-reference',
               sha256=EXPECTED['LDC93S1.wav'], pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
               samples=len(pcm), transcript=transcript, transcript_source='Official LDC sample text',
               target=None)
    manifest = dict(role='Published non-wake speech alignment quality control, never model training/selection',
                    source_sha256=sha(Path(__file__)), records=[row],
                    published_sample_sources=downloads, published_phone_intervals=phones,
                    reference_last_phone=nonsilence[-1],
                    reference_speech_end_seconds=nonsilence[-1]['end_samples']/16000,
                    unannotated_trailing_samples=len(pcm)-phones[-1]['end_samples'],
                    endpoint_tolerance_ms=20, endpoint_tolerance_prespecified=True,
                    method='Same english_us_arpa3.0.0 and single_speaker SQLite alignment; compare terminal non-silence end to published phone boundary;20ms tolerance, no model tuning.',
                    independent_final_audio_accessed=False, wake_scores_accessed=False,
                    limits='One official public TIMIT sample with published phonetic reference. Different sentence, speaker and recording from wake targets. Published interval tail need not cover entire WAV. Agreement on one control is not an endpoint accuracy guarantee, calibrated confidence interval, phonetic quality audit of TTS, or wake-word acceptance.')
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(samples=len(pcm), reference_endpoint_seconds=manifest['reference_speech_end_seconds'])))


if __name__ == '__main__':
    main()
