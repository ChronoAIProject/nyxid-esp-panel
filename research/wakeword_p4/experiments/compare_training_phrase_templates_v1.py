"""Closed-set, known-interval control for cross-engine lexical transfer on training sources."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np

RESULTS = Path('research/wakeword_p4/results')
STEM = '4090-training-phrase-template-transfer'
PLAN = RESULTS / f'{STEM}-plan-v1.json'
SOURCE = Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx')
WORDS = ['Hey Nicks.', 'Hey Nick.', 'Hey Next.', 'Hey Mix.', 'Hey Muse.']
ENGINES = ['piper', 'kokoro', 'qwen']
TIME_BINS = 32
WARP = 4


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def plan():
    lock = RESULTS / '4090-broadcast-channel-bn-training-plan-v1.json'
    original = json.loads(lock.read_bytes())['files']
    index = SOURCE / 'features.json'
    cache = SOURCE / 'features.bin'
    assert sha(index) == original[str(index)] and sha(cache) == original[str(cache)]
    by_voice = defaultdict(dict)
    for r in json.loads(index.read_bytes()):
        key = r['source_pair_key']
        if (r['split'] == 'training' and key['carrier'] == 'bare' and r['variant'] == 0 and
                key['draw'] == 0 and key['scale'] == 1 and r['text'] in WORDS):
            assert key['engine'] == r['engine'] and str(key['conditioning']) == r['conditioning_id']
            group = by_voice[(r['engine'], r['conditioning_id'])]
            assert r['text'] not in group
            group[r['text']] = r
    raw = np.memmap(cache, dtype='<u2', mode='r').reshape(-1, 40)
    selected = []
    for engine in ENGINES:
        voices = [name for (e, name), group in by_voice.items() if e == engine and set(group) == set(WORDS)]
        def rank(name):
            return hashlib.sha256(f'nyx-template-726/{engine}/{name}'.encode()).hexdigest()
        assert len(voices) >= 12
        for number, voice in enumerate(sorted(voices, key=rank)[:12]):
            for word in WORDS:
                row = by_voice[(engine, voice)][word]
                segment = raw[row['offset']:row['offset']+row['frames']]
                selected.append(dict(engine=engine, voice=voice, word=word, source_id=row['source_id'],
                                     pcm_sha256=row['pcm_sha256'], role='reference' if number < 4 else 'query',
                                     offset=row['offset'], frames=row['frames'], lead_frame=row['lead_frame'],
                                     end_sample=row['generated_container_end_sample'],
                                     feature_sha256=hashlib.sha256(segment.tobytes()).hexdigest()))
    assert len({r['pcm_sha256'] for r in selected}) == len(selected) == 180
    write(PLAN, dict(words=WORDS, engines=ENGINES, numpy=np.__version__, time_bins=TIME_BINS, warp=WARP,
                     code_sha256=sha(__file__), index_sha256=sha(index), original_lock_sha256=sha(lock),
                     source_cache_sha256=original[str(cache)], rows=selected,
                     design='Training only. Whole generated bare-word container, frames with right support strictly after lead and at or before end. 32 interpolated frame vectors, per-frame L2 normalization, squared Euclidean local costs. Equal-sized reference banks for each source engine. Predict nearest of 20 references with per-word minimum. Compare fixed linear time to symmetric2 global DTW constrained to four resampled bins. Report every pairing and tie. No streaming search, timing labels from inference, event threshold, optimization or holdout.'))


def representations(rows):
    raw = np.memmap(SOURCE/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    examples = []
    for r in rows:
        segment = raw[r['offset']:r['offset']+r['frames']]
        assert hashlib.sha256(segment.tobytes()).hexdigest() == r['feature_sha256']
        # The zero-based frame's 30-ms analysis window ends at (index+3)*160.
        end_clock = (np.arange(len(segment)) + 3)*160
        inside = (end_clock > (r['lead_frame']+2)*160) & (end_clock <= r['end_sample'])
        sample = np.clip(segment[inside].astype(np.float64)/25.6, 0, 26)
        assert len(sample) >= 2
        x = np.linspace(0, len(sample)-1, TIME_BINS)
        bins = np.stack([np.interp(x, np.arange(len(sample)), band) for band in sample.T], axis=1)
        examples.append(bins / np.maximum(np.linalg.norm(bins, axis=1, keepdims=True), 1e-12))
    return np.stack(examples)


def distances(probe, templates):
    local = np.sum(probe**2, axis=1)[None, :, None] + np.sum(templates**2, axis=2)[:, None, :]
    local = np.maximum(local-2*(probe @ templates.swapaxes(1, 2)), 0)
    linear = local.diagonal(axis1=1, axis2=2).mean(axis=1)
    cost = np.full((len(templates), TIME_BINS+1, TIME_BINS+1), np.inf)
    cost[:, 0, 0] = 0
    for i in range(1, TIME_BINS+1):
        for j in range(max(1, i-WARP), min(TIME_BINS, i+WARP)+1):
            delta = local[:, i-1, j-1]
            cost[:, i, j] = np.minimum(cost[:, i-1, j-1]+2*delta,
                                       np.minimum(cost[:, i-1, j], cost[:, i, j-1])+delta)
    warped = cost[:, -1, -1] / (2*TIME_BINS)
    assert np.all(np.isfinite(warped)) and np.all(warped <= linear+1e-12)
    return linear, warped


def run():
    config = json.loads(PLAN.read_bytes())
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:'+str(PLAN)]) == PLAN.read_bytes()
    assert config['code_sha256'] == sha(__file__) and config['numpy'] == np.__version__
    assert (config['words'], config['engines'], config['time_bins'], config['warp']) == (WORDS, ENGINES, TIME_BINS, WARP)
    rows = config['rows']
    refs = [r for r in rows if r['role'] == 'reference']
    queries = [r for r in rows if r['role'] == 'query']
    bank = representations(refs)
    probes = representations(queries)
    linear, warped = np.stack([distances(q, bank) for q in probes], axis=1)
    assert linear.shape == warped.shape == (120, 60)
    confusions = {}
    ties = []
    for method, table in [('linear', linear), ('dtw', warped)]:
        for i, query in enumerate(queries):
            for engine in ENGINES:
                scores = [min(table[i, j] for j, ref in enumerate(refs)
                              if (ref['engine'], ref['word']) == (engine, word)) for word in WORDS]
                winners = np.flatnonzero(np.asarray(scores) == min(scores))
                if len(winners) > 1:
                    ties.append(dict(query=query['source_id'], engine=engine, method=method,
                                     labels=[WORDS[k] for k in winners]))
                key = '/'.join([method, query['engine'], engine])
                matrix = confusions.setdefault(key, np.zeros((len(WORDS), len(WORDS)), np.int32))
                matrix[WORDS.index(query['word']), winners[0]] += 1
    score_path = RESULTS / f'{STEM}-distances-v1.npz'
    with score_path.open('xb') as stream:
        np.savez_compressed(stream, linear=linear, dtw=warped)
    assert all(np.all(m.sum(axis=1)==8) for m in confusions.values())
    write(RESULTS / f'{STEM}-original-v1.json',
          dict(plan_sha256=sha(PLAN), distances_sha256=sha(score_path), axes=['query', 'reference'],
               query_ids=[r['source_id'] for r in queries], reference_ids=[r['source_id'] for r in refs],
               matrices={k:v.tolist() for k,v in confusions.items()}, ties=ties,
               model_invocations=0, scope=config['design']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    {'plan': plan, 'run': run}[parser.parse_args().action]()
