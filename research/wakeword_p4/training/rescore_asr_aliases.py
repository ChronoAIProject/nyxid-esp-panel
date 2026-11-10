"""Recount ASR text agreement with declared wake aliases also inside commands."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def normalize(text):
    return ' '.join(re.findall(r'[a-z]+', text.lower()))


def aliases(expected):
    prefixes = ['hey nyx', 'hey nix', 'hey nicks', 'hey knicks', 'hey nick s']
    for prefix in prefixes:
        if expected == prefix or expected.startswith(prefix+' '):
            return {p+expected[len(prefix):] for p in prefixes}
    return {expected}


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--details', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = [json.loads(x) for x in a.details.read_text().splitlines()]
counts = Counter(); confusions = Counter()
for row in rows:
    text, expected = normalize(row['transcript']), normalize(row['expected'])
    accepted = aliases(expected) if row['target'] == 'hey_nyx' else {expected}
    counts[row['target']+'/clips'] += 1
    counts[row['target']+'/alias_text_agreements'] += int(text in accepted)
    if row['target'] == 'hey_nyx':
        counts['standalone/clips' if expected == 'hey nyx' else 'command/clips'] += 1
        counts['standalone/alias_text_agreements' if expected == 'hey nyx' else 'command/alias_text_agreements'] += int(text in accepted)
        if text not in accepted:
            confusions[text] += 1
report = dict(role='Training-only ASR recount, preserving command suffix for declared aliases',
              details_sha256=hashlib.sha256(a.details.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              counts=dict(counts), unmatched_positive_transcripts=dict(confusions),
              audio_or_labels_changed=False, wake_scores_computed=False,
              limits='ASR errors are possible; these counts are text consistency, not phonetic truth or wake recognition accuracy. No clips are selected or filtered.')
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
