"""Training-only paired phonetic targets and temporal auxiliary supervision."""
import hashlib
import json

import numpy as np

from matched_confuser_windows import sha

FRAMES, STEPS, VOCAB = 335, 67, 392


class Sampler:
    def __init__(self, path, records, index_sha256, frames):
        meta = json.loads((path/'bank.json').read_bytes())
        self.bank_sha256 = sha(path/'bank.json')
        if (frames != FRAMES or meta['feature_index_sha256'] != index_sha256
                or (meta['frames'], meta['encoder_steps'], meta['vocabulary'], meta['blank']) != (FRAMES, STEPS, VOCAB, 0)
                or sha(path/'bank.npz') != meta['data_sha256']):
            raise ValueError('Phone bank input identity differs')
        with np.load(path/'bank.npz', allow_pickle=False) as data:
            self.features = data['features']
            self.targets = data['posteriors']
        rows = meta['rows']
        count = meta['pair_count']
        if (self.features.shape != (2*count, FRAMES, 40) or self.features.dtype != np.uint16
                or self.targets.shape != (2*count, STEPS, VOCAB) or self.targets.dtype != np.float32
                or not np.isfinite(self.targets).all() or np.any(self.targets < 0)
                or not np.allclose(self.targets.sum(-1), 1, atol=1e-5) or len(rows) != count*2):
            raise ValueError('Invalid phone bank geometry/probability')
        by_key = {(r['path'], r['variant']): r for r in records}
        reserved = {r['sha256'] for r in records if r['split'] != 'training'}
        for i, row in enumerate(rows):
            original = by_key[row['path'], row['variant']]
            if (original['split'] != 'training' or original['sha256'] != row['source_sha256']
                    or original['sha256'] in reserved or original['label'] != row['label']
                    or row['label'] != 1-i%2 or row['pair'] != i//2
                    or hashlib.sha256(self.features[i].tobytes()).hexdigest() != row['feature_sha256']):
                raise ValueError('Phone target changes source labels or includes reserved audio')
        self.rng = np.random.default_rng(431)
        self.order = iter(())
        self.draws = 0
        self.digest = hashlib.sha256()
        self.count = count

    def apply(self, samples):
        ids = np.zeros(len(samples), dtype=np.float32)
        for slot in range(8):
            pair = next(self.order, None)
            if pair is None:
                self.order = iter(self.rng.permutation(self.count))
                pair = next(self.order)
            for index, target in ((slot, pair*2), (32+slot, pair*2+1)):
                samples[index] = self.features[target]
                ids[index] = target+1
            self.digest.update(f'{pair}\n'.encode())
            self.draws += 1
        return ids[:, None]

    def metadata(self):
        return {'bank_sha256': self.bank_sha256, 'pairs': self.count, 'draws': self.draws,
                'schedule_sha256': self.digest.hexdigest(), 'steps': STEPS, 'vocabulary': VOCAB}


def supervisor(model, targets, gain):
    import tensorflow as tf

    temporal = tf.keras.layers.Reshape((STEPS, 60), name='training_phone_sequence')(
        model.get_layer('classifier').input)
    logits = tf.keras.layers.Dense(VOCAB, name='training_phone_logits',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=919))(temporal)
    teacher = tf.constant(targets)
    full = tf.keras.Model(model.input, (model.output, logits))

    @tf.function
    def train(x, y, weights):
        with tf.GradientTape() as tape:
            probability, phones = full(x, training=True)
            binary = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], probability)*weights)
            ids = tf.cast(y[:, 1], tf.int32)
            present = ids > 0
            q = tf.gather(teacher, tf.boolean_mask(ids, present)-1)
            logp = tf.nn.log_softmax(tf.boolean_mask(phones, present), axis=-1)
            divergence = tf.reduce_sum(q*(tf.math.log(tf.maximum(q, 1e-30))-logp), axis=-1)
            # Retain blank supervision while emphasizing the sparse phone spikes.
            frame_weight = .05+.95*(1-q[..., 0])
            phonetic = tf.reduce_sum(frame_weight*divergence)/tf.reduce_sum(frame_weight)
            loss = binary+gain*phonetic
        variables = full.trainable_variables
        model.optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    def audit(features):
        total_weight = total_kl = 0.
        nonblank_total = nonblank_correct = blank_total = blank_correct = 0
        for start in range(0, len(features), 32):
            x = np.clip(features[start:start+32].astype(np.float32)/25.6, 0, 26)
            q = targets[start:start+32]
            logits = np.asarray(full(x, training=False)[1])
            logp = np.asarray(tf.nn.log_softmax(logits, axis=-1))
            divergence = np.sum(q*(np.log(np.maximum(q, 1e-30))-logp), axis=-1)
            weight = .05+.95*(1-q[..., 0])
            total_kl += float(np.sum(weight*divergence))
            total_weight += float(weight.sum())
            labels, prediction = q.argmax(-1), logits.argmax(-1)
            nonblank = labels != 0
            nonblank_total += int(nonblank.sum())
            nonblank_correct += int(((prediction == labels) & nonblank).sum())
            blank_total += int((~nonblank).sum())
            blank_correct += int(((prediction == labels) & ~nonblank).sum())
        return {'weighted_kl': total_kl/total_weight,
                'nonblank_frames': nonblank_total, 'nonblank_correct': nonblank_correct,
                'blank_frames': blank_total, 'blank_correct': blank_correct}

    train.audit = audit
    train.auxiliary_model = full
    return train
