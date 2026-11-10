"""Training-only lexical supervision; exported detector remains one sigmoid.

The binary sampler/loss, source speed, input arrays and runtime graph are
unchanged. An auxiliary six-class loss teaches the shared dense representation
to distinguish the target from exact short confusers. Control weight is zero.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import wave

import numpy as np

CLASSES = ['target', 'hey nick', 'hey next', 'hey nice', 'hey mix', 'hey nexus']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lexical_class(label, texts):
    if label == 1:
        return 0
    names = {re.sub(r'[^a-z ]', '', text.lower()).strip() for text in texts}
    classes = {CLASSES.index(name) for name in names if name in CLASSES[1:]}
    if len(classes) > 1:
        raise ValueError('One source has conflicting lexical labels')
    return next(iter(classes)) if classes else None


def rms_end(pcm):
    """Training-negative auxiliary anchor only; never a phonetic truth label."""
    if not len(pcm):
        raise ValueError('Empty source audio')
    signal = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % 160))
    rms = np.sqrt(np.mean(signal.reshape(-1, 160) ** 2, axis=1))
    active = np.flatnonzero(rms >= max(1., float(rms.max()) * .01))
    if not len(active):
        raise ValueError('Silent source cannot define a lexical auxiliary window')
    return min((int(active[-1]) + 1) * 160, len(pcm)) / 16000


def prepare(args):
    plan = json.loads(args.plan.read_text())
    if sha(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Frozen preparation script differs')
    if sha(args.metadata) != plan['source_metadata_sha256']:
        raise ValueError('Frozen original text metadata differs')
    metadata = defaultdict(list)
    heldout = set()
    for row in json.loads(args.metadata.read_text()):
        if row['split'] != 'training':
            heldout.add(row['sha256'])
        elif row['variant'] == 0:
            metadata[row['sha256']].append(row)
    base = args.data_root / 'head-features-tts-v1'
    if sha(base / 'index.jsonl') != plan['base_feature_index_sha256'] or sha(base / 'jobs-private.json') != plan['base_jobs_sha256']:
        raise ValueError('Frozen base feature/source index differs')
    jobs = {r['id']: r for r in json.loads((base / 'jobs-private.json').read_text())}
    records, counts = [], Counter()
    negative_ends = {}
    for row in map(json.loads, (base / 'index.jsonl').read_text().splitlines()):
        if row['split'] != 'training':
            continue
        aliases = metadata[row['source_sha256']]
        if (not aliases or row['source_sha256'] in heldout or any(
                (r['label'], r['engine']) != (row['label'], row['engine']) for r in aliases)):
            raise ValueError('Original training source role or split differs')
        cls = lexical_class(row['label'], [r['text'] for r in aliases])
        if cls is None:
            continue
        if row['label']:
            anchor = row['keyword_end']
        else:
            source_hash = row['source_sha256']
            if source_hash not in negative_ends:
                source = Path(jobs[row['id']]['source'])
                if sha(source) != source_hash:
                    raise ValueError('Original training source PCM differs')
                with wave.open(str(source), 'rb') as wav:
                    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                        raise ValueError('Expected unmodified mono16kHz PCM16 source')
                    negative_ends[source_hash] = rms_end(np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2'))
            anchor = 2.4 + negative_ends[source_hash]
        low, high = int(np.ceil((anchor + .03) / .08)), int(np.ceil((anchor + .32) / .08))
        if not 25 <= low <= high <= row['frames']:
            raise ValueError('Auxiliary sampling bounds exceed original warmed features')
        records.append({key: row[key] for key in ('id', 'source_sha256', 'features_sha256', 'engine', 'label', 'variant')}
                       | {'class': cls, 'minimum_end': low, 'maximum_end': high})
        counts[(cls, row['engine'])] += 1
    if set(r['class'] for r in records) != set(range(len(CLASSES))):
        raise ValueError('All six lexical training classes are required')
    result = {'scope': 'Training-only auxiliary windows; no classifier scores or source relabeling',
        'classes': CLASSES, 'preparation_plan_sha256': sha(args.plan),
        'source_metadata_sha256': sha(args.metadata), 'base_feature_index_sha256': sha(base / 'index.jsonl'),
        'all_sources_training_only': True, 'heldout_source_hash_disjoint': True,
        'audio_speed_changed': False, 'speaker_playback': False, 'records': records,
        'counts': [{'class': CLASSES[c], 'engine': engine, 'feature_views': count} for (c, engine), count in sorted(counts.items())],
        'unique_source_recordings': len({r['source_sha256'] for r in records}),
        'negative_anchor_policy': 'Original source last10ms RMS frame >=max(peak*.01,1), plus unchanged2.4s lead; auxiliary metadata only, not phonetic truth. No PCM modification.',
        'positive_anchor_policy': 'Unchanged original binary-training keyword_end metadata; not new alignment.',
        'accepted': False}
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'counts')}
                     | {'feature_views': len(records), 'manifest_sha256': sha(args.output)}), flush=True)


class AuxiliarySampler:
    def __init__(self, rows, records, seed):
        self.rng = np.random.default_rng(seed)
        by_id = {r['id']: r for r in rows}
        self.groups = defaultdict(list)
        seen = set()
        for record in records:
            identity = record['id']
            if identity in seen or identity not in by_id:
                raise ValueError('Duplicate or missing auxiliary feature identity')
            seen.add(identity)
            row = by_id[identity]
            if (row['split'] != 'training' or row['kind'] != 'tts'
                    or row['_feature_directory'] != 'head-features-tts-v1'
                    or any(row[k] != record[k] for k in ('source_sha256', 'features_sha256', 'engine', 'label', 'variant'))
                    or type(record['class']) is not int or not 0 <= record['class'] < len(CLASSES)
                    or bool(row['label']) != (record['class'] == 0)
                    or not 25 <= record['minimum_end'] <= record['maximum_end'] <= row['frames']):
                raise ValueError('Auxiliary role, identity, label or bounds differ')
            self.groups[(record['class'], row['engine'])].append((row, record))
        self.engines = {c: sorted(k[1] for k in self.groups if k[0] == c) for c in range(len(CLASSES))}
        if any(not engines for engines in self.engines.values()):
            raise ValueError('Every auxiliary class needs training features')

    def batch(self, count=64):
        examples, labels = [], []
        for _ in range(count):
            cls = int(self.rng.integers(len(CLASSES)))
            engine = self.rng.choice(self.engines[cls])
            group = self.groups[(cls, engine)]
            row, record = group[int(self.rng.integers(len(group)))]
            end = int(self.rng.integers(record['minimum_end'], record['maximum_end'] + 1))
            examples.append(row['array'][end - 16:end])
            labels.append(cls)
        return np.asarray(examples, np.float32)[:, :, :, None] / 128, np.asarray(labels, np.int32)


def train(args):
    import tensorflow as tf
    from train_head import Sampler, export, feature_directories, load_rows, weighted_loss

    plan = json.loads(args.plan.read_text())
    root = Path(__file__).parent
    for name, expected in plan['script_files'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen training/helper scripts differ')
    if sha(args.initial_weights) != plan['initial_weights_sha256'] or sha(args.auxiliary_manifest) != plan['auxiliary_manifest_sha256']:
        raise ValueError('Frozen initialization or auxiliary manifest differs')
    auxiliary = json.loads(args.auxiliary_manifest.read_text())
    if (not auxiliary['all_sources_training_only'] or not auxiliary['heldout_source_hash_disjoint']
            or auxiliary['classes'] != CLASSES or auxiliary['audio_speed_changed']):
        raise ValueError('Auxiliary source roles or class order differs')
    for name, expected in plan['feature_indexes'].items():
        if sha(args.data_root / name / 'index.jsonl') != expected:
            raise ValueError('Frozen main training feature index differs')
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(plan['recipe']['seed'])
    started = time.monotonic()
    extra = plan['extra_training_features']
    rows = load_rows('training', args.data_root, plan['model_sha256'], extra)
    heldout = {row['source_sha256'] for name in feature_directories(extra)
        for row in map(json.loads, (args.data_root / name / 'index.jsonl').read_text().splitlines())
        if row['split'] != 'training'}
    if heldout.intersection(row['source_sha256'] for row in rows):
        raise ValueError('Main training source overlaps held-out features')
    sampler = Sampler(rows, plan['recipe']['seed'])
    aux_sampler = AuxiliarySampler(rows, auxiliary['records'], plan['recipe']['seed'] + 2000000)
    representatives = Sampler([r for r in rows if r['_feature_directory'] in
        ('head-features-tts-v1', 'head-features-negatives-v1')], plan['recipe']['seed'] + 1)
    examples = np.concatenate([representatives.batch()[0] for _ in range(4)])
    model = tf.keras.models.load_model(args.initial_weights)
    layer = tf.keras.layers.Dense(len(CLASSES), name='training_only_lexical_logits',
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=plan['recipe']['seed'] + 1))
    lexical = tf.keras.Model(model.input, layer(model.layers[-2].output))
    weight = plan['auxiliary_loss_weight']
    if weight not in (0, .2):
        raise ValueError('Expected prespecified zero/.2 matched experiment')
    variables = model.trainable_variables + (layer.trainable_variables if weight else [])
    optimizer = tf.keras.optimizers.Adam(plan['recipe']['learning_rate'])

    @tf.function
    def step(x, y, auxiliary_x, auxiliary_y):
        with tf.GradientTape() as tape:
            binary = weighted_loss(y, model(x, training=True), plan['recipe'])
            if weight:
                logits = lexical(auxiliary_x, training=True)
                loss_aux = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(auxiliary_y, logits, from_logits=True))
            else:
                loss_aux = tf.constant(0.)
            loss = binary + weight * loss_aux
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return binary, loss_aux

    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'representatives-training-only.npy', examples)
    summary = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'plan': plan,
        'parameters': model.count_params(), 'auxiliary_parameters_training_only': layer.count_params(),
        'training_sources': len(rows), 'training_frames': sum(r['frames'] for r in rows),
        'auxiliary_manifest_sha256': sha(args.auxiliary_manifest),
        'initial_weights_sha256': sha(args.initial_weights), 'feature_inputs': {}, 'checkpoints': []}
    for name in feature_directories(extra):
        base = args.data_root / name
        summary['feature_inputs'][name] = {'index_sha256': sha(base / 'index.jsonl'),
            'identity': json.loads((base / 'identity.json').read_text()),
            'completion': json.loads((base / 'complete.json').read_text())}
    (args.output / 'training.json').write_text(json.dumps(summary, indent=2) + '\n')
    main_hash, aux_hash = hashlib.sha256(), hashlib.sha256()
    recipe = plan['recipe']
    for number in range(1, recipe['steps'] + 1):
        fraction = number / recipe['steps']
        factor = recipe['learning_rate_eighty_percent_factor'] if fraction > .8 else (
            recipe['learning_rate_half_factor'] if fraction > .5 else 1)
        optimizer.learning_rate.assign(recipe['learning_rate'] * factor)
        x, y = sampler.batch()
        ax, ay = aux_sampler.batch()
        for value in (x, y):
            main_hash.update(value.tobytes())
        for value in (ax, ay):
            aux_hash.update(value.tobytes())
        binary, loss_aux = step(x, y, ax, ay)
        if number % 1000 == 0:
            print(json.dumps({'step': number, 'binary_loss': float(binary), 'auxiliary_loss': float(loss_aux),
                              'seconds': time.monotonic() - started}), flush=True)
        if number in recipe['checkpoints']:
            output = args.output / f'step-{number:05d}'
            output.mkdir()
            report = export(model, output, examples)
            summary['checkpoints'].append({'step': number, **report})
            print(json.dumps({'phase': 'exported', 'step': number, **report}), flush=True)
    lexical.save(args.output / 'training-only-auxiliary.keras')
    summary.update(seconds=time.monotonic() - started, main_batches_sha256=main_hash.hexdigest(),
                   auxiliary_batches_sha256=aux_hash.hexdigest(), accepted=False)
    (args.output / 'training.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'phase': 'completed', 'seconds': summary['seconds'],
        'main_batches_sha256': summary['main_batches_sha256'], 'auxiliary_batches_sha256': summary['auxiliary_batches_sha256']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'train'):
        command = sub.add_parser(name)
        command.add_argument('--data-root', type=Path, required=True)
        command.add_argument('--plan', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'prepare':
            command.add_argument('--metadata', type=Path, required=True)
        else:
            command.add_argument('--initial-weights', type=Path, required=True)
            command.add_argument('--auxiliary-manifest', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
