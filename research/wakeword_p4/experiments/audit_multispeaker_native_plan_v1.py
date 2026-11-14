"""Independent schedule/pronunciation audit; zero synthesis/model calls."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from piper_phonemize import phonemize_espeak

R = Path('research/wakeword_p4/results')


def main():
    path = R / '4090-multispeaker-coverage-native-plan-v1.json'
    plan = json.loads(path.read_text())
    models = {m['voice']: m for m in plan['models']}
    assert set(models) == {'en_GB-aru-medium', 'en_US-arctic-medium'}
    seen = set()
    ids = set()
    cells = {a: Counter() for a in ['narrow', 'broad']}
    speaker_cells = {a: defaultdict(Counter) for a in cells}
    for row in plan['requests']:
        assert row['role'] == 'training'
        assert row['label'] == int(row['target'] == 'hey_nyx')
        key = (row['voice'], row['speaker_id'], row['text'], row['length_scale'], row['draw'])
        assert key not in seen and row['id'] not in ids
        seen.add(key)
        ids.add(row['id'])
        model = models[row['voice']]
        config = json.loads(Path(model['config_path']).read_text())
        assert model['speaker_id_map'][row['speaker_name']] == row['speaker_id']
        expected_phones = phonemize_espeak(row['text'], config['espeak']['voice'])
        assert expected_phones == [row['phonemes']]
        symbols = ['^', '_'] + [v for phone in row['phonemes'] for v in (phone, '_')] + ['$']
        expected_tokens = [v for symbol in symbols for v in config['phoneme_id_map'][symbol]]
        assert row['token_ids'] == expected_tokens
        narrow = row['speaker_name'] in plan['narrow_speakers'][row['voice']]
        broad = row['draw'] < 36 // model['num_speakers']
        assert set(row['arms']) == ({'narrow'} if narrow else set()) | ({'broad'} if broad else set())
        assert 0 <= row['draw'] < (12 if narrow else 36 // model['num_speakers'])
        for arm in row['arms']:
            cells[arm][(row['voice'], row['text'], row['length_scale'])] += 1
            speaker_cells[arm][row['voice']][row['speaker_name']] += 1
    for arm in cells:
        assert len(cells[arm]) == 120 and set(cells[arm].values()) == {36}
        assert sum(cells[arm].values()) == 4320
    assert len(ids) == 7740
    for voice, model in models.items():
        ranked = sorted(model['speaker_id_map'], key=lambda speaker: hashlib.sha256(
            ('nyx-multispeaker-coverage-v1:' + voice + ':' + speaker).encode()).hexdigest())
        assert plan['narrow_speakers'][voice] == ranked[:3]
        assert set(speaker_cells['broad'][voice]) == set(model['speaker_id_map'])
        assert set(speaker_cells['narrow'][voice]) == set(ranked[:3])
    report = dict(schema='multispeaker-native-plan-independent-audit-v1',
                  plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  unique_requests=len(ids), shared_requests=sum(len(r['arms']) == 2 for r in plan['requests']),
                  arm_requests={a: sum(c.values()) for a, c in cells.items()},
                  equal_model_text_speed_cell_count=36, cells_per_arm=120,
                  token_and_speaker_schedule_checks_passed=True,
                  synthesis_calls=0, detector_invocations=0,
                  limits='Prospective identity/token/schedule consistency only; not actual saved waveform quality or independent source certification.')
    with (R / '4090-multispeaker-native-plan-independent-audit-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
