"""Exercise the new gate on consumed SLR83 outputs, never reclassify data roles."""
import json
from pathlib import Path
from synthetic_output_quality_v1 import POLICY, audit_output, sha


def main():
    root = Path('.local/wakeword/slr83-development-v1')
    source_root = Path('.local/wakeword/piper-slr83-development-source-v1')
    source = json.loads((source_root / 'source.json').read_text())
    original = json.loads((root / 'report.json').read_text())
    assert sha(source_root / 'source.json') == original['source_sha256']
    assert sha(root / 'manifest.jsonl') == original['manifest_sha256']
    assert sha(Path('research/wakeword_p4/training/generate_slr83_development.py')) == original['script_sha256']
    for name, item in source['files'].items():
        assert sha(source_root / name) == item['sha256']
        assert (source_root / name).stat().st_size == item['size_bytes']
    phrases = dict(hey_nyx='Hey Nicks.', hi_esp='Hi E S P.',
                   hey_nick='Hey Nick.', hey_next='Hey Next.', hey_mix='Hey Mix.')
    details = []
    for line in (root / 'manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert row['role'] == 'slr83-development-only' and row['split'] == 'validation'
        assert row['text'] == phrases[row['target']]
        name = f"en_GB-{row['voice']}-{source['voices'][row['voice']]}.onnx.json"
        config = json.loads((source_root / row['voice'] / name).read_text())
        mapping = config['phoneme_id_map']
        tokens = list(mapping['^']) + list(mapping['_'])
        for phone in row['phonemes']:
            tokens += list(mapping[phone]) + list(mapping['_'])
        tokens += list(mapping['$'])
        assert tokens == row['token_ids'] and row['phonemes']
        assert row['native_durations'] is None and row['duration_limit']
        result = audit_output(root, row)
        details.append(dict(path=row['path'], **result))
    assert len(details) == original['clips'] == 30
    report = dict(policy=POLICY, source_sha256=sha(source_root / 'source.json'),
                  source_revision=source['revision'], original_manifest_sha256=sha(root / 'manifest.jsonl'),
                  original_generator_versions=original['versions'],
                  validator_sha256=sha(Path(__file__).with_name('synthetic_output_quality_v1.py')),
                  script_sha256=sha(Path(__file__)), clips=len(details),
                  passed=sum(r['passed'] for r in details), details=details,
                  new_audio_generated=False, detector_scored=False, data_changed=False,
                  limits='Consumed historical outputs only; no new development or acceptance evidence. '
                  'Checks bind complete saved native output to PCM, text/phone token ledger and source bytes. '
                  'They cannot prove semantic full-phrase generation, acoustic phonemes, physical speaker '
                  'independence or original engine-return completeness beyond the authenticated generator. '
                  'Unavailable native token durations remain null; no human reference fabricated.')
    output = Path('research/wakeword_p4/results/4090-saved-synthetic-output-quality-v1.json')
    with output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(clips=report['clips'], passed=report['passed'])))


if __name__ == '__main__':
    main()
