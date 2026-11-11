"""Opt-in replacement of half the v37 phrase slots with complete old sources.

Superseded before real-pack execution by GPU v40's frozen CompleteCoverage
implementation at 911937b (seed127726). This unused seed137726 prototype is
retained for provenance only; do not integrate it into the running experiment.
Coordination: https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13

Call after consuming the original baseline128 and FullSourceBatch.first80().
This module never owns that generator, its RNG, loss, model or optimizer.
Only the returned x changes; y/weights and unselected input slots are preserved.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PACK_SHA = 'bb01f3219db10f31e1095af4912c1807ec38b0b3c640c4676186c1b77a620aa1'
SOURCES_SHA = '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
RECORDS_SHA = '757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090'
ENGINES = ('kokoro', 'macos-say', 'openai-tts', 'piper', 'qwen')
NEGATIVES = ('hey_nick', 'hey_next', 'hey_mix')
TARGETS = {'hey nyx': 'hey_nyx', 'hey nix': 'hey_nyx', 'hey nicks': 'hey_nyx',
           'hey nick': 'hey_nick', 'hey next': 'hey_next', 'hey mix': 'hey_mix'}
SLOTS = tuple(range(16))+tuple(range(32, 56))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_groups(sources, records, labels):
    by_source = {}
    for row in records:
        by_source.setdefault(row['source_index'], []).append(row)
    groups = {}
    for source_index, source in enumerate(sources):
        aliases = source['aliases']
        label = aliases[0]['label']
        engine = aliases[0]['engine']
        if not all((a['label'], a['engine'], a['sha256']) == (label, engine, source['source_sha256']) for a in aliases):
            raise ValueError('Conflicting canonical source aliases')
        targets = {TARGETS[a['phrase']] for a in aliases if a['phrase'] in TARGETS}
        if not targets:
            continue  # Retain Nexus/Hi ESP in the original pack, outside this arm.
        if len(targets) != 1:
            raise ValueError('Ambiguous phrase class for exact source aliases')
        target, = targets
        if label != int(target == 'hey_nyx'):
            raise ValueError('Existing label disagrees with intended phrase class')
        rows = by_source[source_index]
        if sorted(r['variant'] for r in rows) != [0, 1, 2]:
            raise ValueError('Canonical source requires all three original variants')
        indices = []
        for row in sorted(rows, key=lambda r: r['variant']):
            index = row['pack_index']
            if row['source_sha256'] != source['source_sha256'] or labels[index] != label:
                raise ValueError('Array index/source/label mismatch')
            indices.append(index)
        # Tokens remain engine-qualified by the enclosing engine group. Blends
        # and conditions are metadata families, not independent human voices.
        voice = tuple(sorted({str(v) for a in aliases for v in a['speakers']}))
        if not voice:
            raise ValueError('Empty source voice provenance')
        groups.setdefault(target, {}).setdefault(engine, {}).setdefault(voice, {})[source_index] = tuple(indices)
    if set(groups) != {'hey_nyx', *NEGATIVES} or any(set(engines) != set(ENGINES) for engines in groups.values()):
        raise ValueError('Every phrase class must cover all five engines')
    return groups


def sampling_tables(groups):
    """Cache canonical order once, keeping per-step work independent of corpus size."""
    return {target: {engine: tuple(tuple(voices[voice][source] for source in sorted(voices[voice]))
                                  for voice in sorted(voices))
                     for engine, voices in engines.items()} for target, engines in groups.items()}


class OldSourceReplacement:
    def __init__(self, root, seed=137726):
        root = Path(root)
        report = json.loads((root/'report.json').read_bytes())
        expected = {'training.npz': PACK_SHA, 'sources.json': SOURCES_SHA, 'records.json': RECORDS_SHA}
        if any(sha(root/name) != digest for name, digest in expected.items()):
            raise ValueError('Require frozen complete-container candidate data')
        if (report['pack_sha256'] != PACK_SHA or report['training_run'] or report['model_scored']
                or report['exact_zero_exclusion_used']):
            raise ValueError('Require original source-only candidate pack')
        with np.load(root/'training.npz', allow_pickle=False) as bundle:
            self.features, self.labels = bundle['x'], bundle['y']
        if (self.features.shape != (7782, 150, 40) or self.features.dtype != np.uint16
                or self.labels.shape != (7782,) or self.labels.dtype != np.float32):
            raise ValueError('Frozen candidate arrays differ')
        sources = json.loads((root/'sources.json').read_bytes())
        records = json.loads((root/'records.json').read_bytes())
        if len(sources) != 2594 or [r['pack_index'] for r in records] != list(range(7782)):
            raise ValueError('Frozen source/array correspondence differs')
        self.groups = build_groups(sources, records, self.labels)
        self.tables = sampling_tables(self.groups)
        self.features.setflags(write=False);self.labels.setflags(write=False)
        self.rng = np.random.default_rng(seed)
        self.last_indices = None
        self.report = {'role': 'optional-complete-source-half-phrase-replacement',
            'source_sha256': sha(Path(__file__)), 'pack_sha256': PACK_SHA,
            'sources_sha256': SOURCES_SHA, 'records_sha256': RECORDS_SHA, 'seed': seed,
            'replacement_slots': list(SLOTS), 'new_positive_slots': 16, 'new_negative_slots': 24,
            'policy': 'Consume original v37 draws first. Replace only fixed40 phrase slots with independent RNG. Equal negative class; then equal engine, voice-token family, canonical WAV hash, and variant. Original labels/weights, other40 phrase slots and48 ambient slots retained. No added loss/head/optimizer.',
            'groups': {target: {engine: {'voice_token_families': len(voices),
                'canonical_sources': sum(len(ids) for ids in voices.values()),
                'variants': sum(len(rows) for ids in voices.values() for rows in ids.values())}
                for engine, voices in engines.items()} for target, engines in self.groups.items()},
            'limits': ['Optional data helper, not a registered training run or recognition result.',
                'Source labels are intended texts; aliases, blends and conditions are not independent humans.',
                'Calling after original draws and preserving baseline RNG must also be verified at the training integration.']}

    def sample_indices(self):
        selected = []
        for slot in SLOTS:
            target = 'hey_nyx' if slot < 16 else NEGATIVES[int(self.rng.integers(3))]
            engine = ENGINES[int(self.rng.integers(len(ENGINES)))]
            voices = self.tables[target][engine]
            family = voices[int(self.rng.integers(len(voices)))]
            variants = family[int(self.rng.integers(len(family)))]
            selected.append(variants[int(self.rng.integers(3))])
        return np.asarray(selected, dtype=np.int64)

    def replace(self, x, y, weights):
        if (x.shape != (128, 150, 40) or x.dtype != np.float32
                or y.shape != (128, 1) or weights.shape != (128,)):
            raise ValueError('Require original128 baseline batch')
        if not np.array_equal(y[:80, 0], np.asarray([1]*32+[0]*48)):
            raise ValueError('Require original32-positive/48-confusing phrase layout')
        indices = self.sample_indices()
        expected = np.asarray([1]*16+[0]*24)
        if not np.array_equal(self.labels[indices], expected):
            raise ValueError('Replacement source label differs from baseline slot')
        result = x.copy()
        result[list(SLOTS)] = np.clip(self.features[indices].astype(np.float32)/25.6, 0, 26)
        self.last_indices = indices
        return result
