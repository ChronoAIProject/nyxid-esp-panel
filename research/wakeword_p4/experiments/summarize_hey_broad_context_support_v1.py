"""Enumerate legal training ends from metadata; no features, scores or RNG draws."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('research/wakeword_p4')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    bindings = {}
    targets = []
    for target in ('hey_willow', 'hey_nyx'):
        base = Path('.local/wakeword/hey-broad-training-features-v1') / target
        path = base / 'features.json'
        provenance = base / 'target-provenance.json'
        prov = json.loads(provenance.read_text())
        assert sha(path) == prov['files']['features.json']
        bindings[str(path)] = sha(path)
        bindings[str(provenance)] = sha(provenance)
        groups = collections.defaultdict(collections.Counter)
        for r in json.loads(path.read_text()):
            assert r['split'] == 'training'
            assert r['label'] == int(r['target_id'] == target)
            n, lead, end = r['frames'], r['lead_frame'], r['keyword_frame']
            assert n >= 225
            g = groups[(r['engine'], r['label'])]
            g['rows'] += 1
            if r['label']:
                ends = set(range(end + 3, end + 28))
                assert all(225 <= e <= n and e - 225 <= lead for e in ends)
                # Full generated phrase support means the entire metadata interval
                # [lead, keyword) lies in the context, not a phonetic assertion.
                full = set(range(max(225, end), min(n, lead + 225) + 1))
                assert ends <= full
                g['legal_full_phrase_context_ends'] += len(full)
                g['positive_supervised_ends'] += len(ends)
                g['full_phrase_ends_outside_positive_support'] += len(full - ends)
                g['post_phrase_container_ends'] += max(0, n - end)
                g['post_phrase_ends_beyond_positive_support'] += max(0, n - end - 27)
                if r['engine'] in ('piper', 'kokoro'):
                    assert end - 35 >= 225
                    g['prefix_negative_ends'] += 21
            else:
                lo = max(225, lead - 25)
                assert lo <= n
                g['ordinary_negative_legal_ends_before_replacement'] += n - lo + 1
        targets.append(dict(target=target, groups=[dict(engine=e, label=l, **dict(v))
                       for (e, l), v in sorted(groups.items())]))
    sources = [Path(__file__), ROOT/'training/train_stream.py',
               ROOT/'experiments/audit_hey_broad_batches_v1.py',
               ROOT/'experiments/schedule_hey_broad_v1.py']
    bindings.update({str(p): sha(p) for p in sources})
    return dict(schema='hey-broad-context-support-enumeration-v1', files=bindings,
                targets=targets, model_invocations=0,
                limits='Training metadata legal-end enumeration only; counts sum per row, not unique source or sampling probabilities or actual 10000-step visits. Ordinary negatives are drawn48 then prefix replaces16; discarded draws are not supervision. Ambient windows not enumerated here. Positive support is keyword+3..27; prefix negatives keyword-35..-15 only Piper/Kokoro. Metadata intervals are generated proxies, not phonetic truth. No heldout access, scoring, threshold selection, causal diagnosis or new training authorization. Missing full-phrase contexts do not imply they should all carry positive labels; late contexts may require negative supervision. Prior v43 full-container intervention failed and is not repeated.')

if __name__ == '__main__':
    result = run()
    out = ROOT/'results/4090-hey-broad-context-support-enumeration-v1.json'
    with out.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['targets'], indent=2))
