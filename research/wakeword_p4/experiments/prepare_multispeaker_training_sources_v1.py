"""Authenticate ARU/Arctic for training review only; no synthesis or scoring."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

R = Path('research/wakeword_p4/results')
OUT = Path('.local/wakeword/multispeaker-training-source-review-v1')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py', '--expected-gib', '0.3'], check=True)
    catalog_path = R / '4090-hey-voice-coverage-catalog-v1.json'
    catalog = json.loads(catalog_path.read_text())
    configs = json.loads((R / '4090-hey-coverage-piper-config-metadata-v1.json').read_text())
    cards = json.loads((R / '4090-hey-coverage-source-cards-v1.json').read_text())
    protected = json.loads((R / '4090-hey-coverage-protected-family-policy-v1.json').read_text())['protected_families']
    OUT.mkdir(exist_ok=True)
    rows = []
    for voice in ['en_GB-aru-medium', 'en_US-arctic-medium']:
        family = voice.rsplit('-', 1)[0]
        assert family not in protected
        entry = next(e for e in catalog['english_entries'] if e['key'] == voice)
        config = next(e for e in configs['configs'] if e['voice'] == voice)
        config_raw = Path(config['path']).read_bytes()
        assert digest(config_raw) == config['sha256']
        card_path = Path('.local/wakeword/hey-coverage-source-cards-v1') / (voice + '.md')
        card = card_path.read_bytes()
        original_card = next(e for e in cards['cards'] if e['voice'] == voice)
        assert digest(card) == original_card['sha256']
        assert b'Finetuned from U.S. English lessac' in card
        relative = next(p for p in entry['files'] if p.endswith('.onnx'))
        base = 'https://huggingface.co/rhasspy/piper-voices/'
        pointer_url = base + 'raw/' + catalog['revision'] + '/' + relative
        pointer = urllib.request.urlopen(pointer_url, timeout=60).read()
        expected = re.search(rb'oid sha256:([a-f0-9]{64})', pointer).group(1).decode()
        size = int(re.search(rb'\nsize (\d+)', pointer).group(1))
        dest = OUT / (voice + '.onnx')
        url = base + 'resolve/' + catalog['revision'] + '/' + relative
        if not dest.exists():
            temporary = dest.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('xb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            raw = temporary.read_bytes()
            assert len(raw) == size and digest(raw) == expected
            temporary.rename(dest)
        raw = dest.read_bytes()
        assert len(raw) == size == entry['files'][relative]['size_bytes']
        assert digest(raw) == expected
        assert hashlib.md5(raw).hexdigest() == entry['files'][relative]['md5_digest']
        pointer_path = OUT / (voice + '.lfs.txt')
        if pointer_path.exists():
            assert pointer_path.read_bytes() == pointer
        else:
            pointer_path.write_bytes(pointer)
        rows.append(dict(voice=voice, intended_role='training review only', model_path=str(dest),
                         bytes=size, sha256=expected, url=url, revision=catalog['revision'],
                         pointer_path=str(pointer_path), pointer_sha256=digest(pointer),
                         config_path=config['path'], config_sha256=digest(config_raw),
                         card_path=str(card_path), card_sha256=digest(card),
                         num_speakers=json.loads(config_raw)['num_speakers'],
                         speaker_id_map=json.loads(config_raw)['speaker_id_map'],
                         lineage='Documented Lessac parent already in training; not an independent holdout.'))
    report = dict(schema='multispeaker-training-source-review-v1',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  catalog_sha256=digest(catalog_path.read_bytes()), models=rows,
                  synthesis_authorized=False, training_started=False, detector_invocations=0,
                  limits='Authenticated bytes and declared speaker identities only. Not physical-speaker independence or comprehensive historical role admission. Need original metadata overlap review and a pushed material lock before synthesis. No development or acceptance source reclassification.')
    with (R / '4090-multispeaker-training-source-review-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(authenticated_models=len(rows), declared_speakers=sum(r['num_speakers'] for r in rows))))


if __name__ == '__main__':
    main()
