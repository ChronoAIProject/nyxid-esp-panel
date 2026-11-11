"""Post-audit sensitivity to spacing only; never alter source transcripts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--details', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('Fresh report required')
counts = Counter()
def norm(text):
    return ' '.join(re.findall('[a-z]+', text.lower()))
for row in map(json.loads, a.details.read_text().splitlines()):
    expected = norm(row['expected'])
    accepted = {expected}
    if row['label']:
        prefixes = ['hey nyx','hey nix','hey nicks','hey knicks','hey nick s']
        for prefix in prefixes:
            if expected == prefix or expected.startswith(prefix+' '):
                accepted = {s+expected[len(prefix):] for s in prefixes}
                break
    group = 'positive' if row['label'] else 'negative'
    counts[group+'/clips'] += 1
    counts[group+'/space_insensitive_agreements'] += int(norm(row['transcript']).replace(' ','') in {s.replace(' ','') for s in accepted})
    counts[group+'/strict_agreements'] += row['ctc_agreement']
    counts[group+'/whisper_agreements'] += row['whisper_agreement']
report = dict(role='Post-audit token-spacing sensitivity; original strict counts retained',
    details_sha256=hashlib.sha256(a.details.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    counts=dict(counts), normalization='Original target aliases preserving command suffix; remove spaces only, no new lexical aliases or edit-distance tolerance',
    audio_or_labels_changed=False, final_data_used=False,
    limits='Spacing explains some CTC disagreement. Remaining disagreement is not proof of mispronunciation; neither normalization supplies phone boundaries or wake recall.')
a.output.write_text(json.dumps(report,indent=2)+'\n')
