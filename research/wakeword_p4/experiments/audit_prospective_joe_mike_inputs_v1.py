"""Preflight pinned rdh-lineage Joe/Mike inputs; never synthesize or score audio."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

from piper_phonemize import phonemize_espeak

INVENTORY = Path('research/wakeword_p4/results/4090-prospective-joe-mike-source-plan-v1.json')
PHRASES = (
    ('hey_nyx', 'Hey Nicks.'), ('hi_esp', 'Hi E S P.'),
    ('hey_nick', 'Hey Nick.'), ('hey_next', 'Hey Next.'),
    ('hey_mix', 'Hey Mix.'), ('hey_nexus', 'Hey Nexus.'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = json.loads(INVENTORY.read_text())
    rows = []
    for model in inventory['models']:
        assert sha(Path(model['model_path'])) == model['expected_sha256']
        config_path = Path(model['config_path'])
        assert sha(config_path) == model['config_sha256']
        config = json.loads(config_path.read_text())
        assert config['num_speakers'] == 1
        ids = config['phoneme_id_map']
        for target, text in PHRASES:
            sentences = phonemize_espeak(text, config['espeak']['voice'])
            assert len(sentences) == 1
            phones = sentences[0]
            assert phones and all(phone in ids for phone in phones)
            tokens = list(ids['^']) + list(ids['_'])
            for phone in phones:
                tokens += list(ids[phone]) + list(ids['_'])
            tokens += list(ids['$'])
            assert tokens and all(isinstance(i, int) and i >= 0 for i in tokens)
            rows.append(dict(model=model['voice'], target=target, text=text,
                             espeak_voice=config['espeak']['voice'],
                             phonemes=phones, token_ids=tokens))
    report = dict(inventory_sha256=sha(INVENTORY), script_sha256=sha(Path(__file__)),
                  versions={'piper-phonemize': importlib.metadata.version('piper-phonemize')},
                  inputs=rows, all_symbols_mapped=True, audio_generated=False,
                  source_roles_reserved=False, wake_models_scored=False,
                  limits='Input metadata only, not acoustic phonetic truth, phrase completeness or speaker independence. This prospective phrase list is not a frozen development/acceptance protocol.')
    out = Path('research/wakeword_p4/results/4090-prospective-joe-mike-input-preflight-v1.json')
    with out.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(dict(inputs=len(rows), all_symbols_mapped=True, audio_generated=False)))


if __name__ == '__main__':
    main()
