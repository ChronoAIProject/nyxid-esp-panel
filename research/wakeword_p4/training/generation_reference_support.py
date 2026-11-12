"""Deterministic endpoint substitution; callers retain original RNG draws."""
import hashlib
import json
from pathlib import Path


def choose_endpoint(original, interval, enabled):
    """Keep controls/fallbacks unchanged; use nearest legal endpoint otherwise."""
    if not interval['count']:
        return original, 'unsupported_fallback'
    if not enabled:
        return original, 'control'
    chosen = min(interval['last'], max(interval['first'], original))
    return chosen, 'unchanged_supported' if chosen == original else 'substituted'


class GenerationReferenceSupport:
    def __init__(self, path, expected_sha256):
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError('Changed reference support ledger')
        self.report = json.loads(payload)
        self.rows = {r['feature_index']: r for r in self.report['rows']}
        if len(self.rows) != 63300:
            raise ValueError('Wrong positive reference population')

    def choose(self, feature_index, row, original, enabled):
        support = self.rows[feature_index]
        if support['source_sha256'] != row['sha256'] or support['variant'] != row['variant']:
            raise ValueError('Reference source mismatch')
        return choose_endpoint(original, support['target_endpoints'], enabled)
