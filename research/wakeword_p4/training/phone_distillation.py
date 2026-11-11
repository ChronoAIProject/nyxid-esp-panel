"""Training-only paired phonetic targets and temporal auxiliary supervision."""
import hashlib
import json
from collections import Counter

import numpy as np

from matched_confuser_windows import sha

FRAMES, STEPS, VOCAB = 335, 67, 392


class Sampler:
    def __init__(self, path, records, index_sha256, frames, objective='joint'):
        meta = json.loads((path/'bank.json').read_bytes())
        self.bank_sha256 = sha(path/'bank.json')
        if (frames != FRAMES or meta['feature_index_sha256'] != index_sha256
                or (meta['frames'], meta['encoder_steps'], meta['vocabulary'], meta['blank']) != (FRAMES, STEPS, VOCAB, 0)
                or sha(path/'bank.npz') != meta['data_sha256']):
            raise ValueError('Phone bank input identity differs')
        with np.load(path/'bank.npz', allow_pickle=False) as data:
            self.features = data['features']
            self.targets = data['posteriors']
            if objective == 'ctc' and 'teacher_posteriors' not in data:
                raise ValueError('CTC requires retained full-resolution teacher frames')
            teacher_frames = data['teacher_posteriors'] if objective == 'ctc' else None
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
        self.sequence_metadata = None
        if objective == 'ctc':
            from prepare_phone_bank import align_posteriors
            from phone_sequences import pack_teacher_targets
            if (meta.get('retained_teacher_frames') != 168 or teacher_frames.shape != (2*count, 168, VOCAB)
                    or teacher_frames.dtype != np.float32 or meta.get('teacher_frame_stride_samples') != 320
                    or meta.get('teacher_frame_support_samples') != 400
                    or not np.array_equal(np.stack([align_posteriors(row) for row in teacher_frames]), self.targets)):
                raise ValueError('Full-resolution teacher identity or aligned targets differ')
            self.targets, sequences = pack_teacher_targets(teacher_frames, STEPS)
            self.sequence_metadata = {
                'sequence_sha256': hashlib.sha256(json.dumps(sequences, separators=(',', ':')).encode()).hexdigest(),
                'teacher_posteriors_sha256': hashlib.sha256(teacher_frames.astype('<f4', copy=False).tobytes()).hexdigest(),
                'length_histogram': dict(sorted(Counter(map(len, sequences)).items())),
                'encoding': 'Full168 greedy CTC collapse; nonempty labels padded by blank zero to 67 steps.'}
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
                'schedule_sha256': self.digest.hexdigest(), 'steps': STEPS, 'vocabulary': VOCAB,
                'sequence': self.sequence_metadata}


def balance_encoder_gradients(binary, phone, encoder_indices, gain, ratio_cap):
    """Limit only the scaled auxiliary encoder gradient; preserve both heads."""
    import tensorflow as tf

    bn = tf.linalg.global_norm([binary[i] for i in encoder_indices])
    pn = gain*tf.linalg.global_norm([phone[i] for i in encoder_indices])
    scale = tf.where(pn > 0, tf.minimum(1., tf.math.divide_no_nan(ratio_cap*bn, pn)), 1.)
    mixed = [b+gain*p*(scale if i in encoder_indices else 1.)
             for i, (b, p) in enumerate(zip(binary, phone, strict=True))]
    return mixed, scale, tf.math.divide_no_nan(scale*pn, bn)


def supervisor(model, targets, gain, objective='joint', encoder_ratio_cap=None, background_gain=None,
               positive_gain=None, intent_gain=None):
    import tensorflow as tf

    if objective not in ('joint', 'conditional', 'ctc'):
        raise ValueError('Unknown phone objective')
    if encoder_ratio_cap is not None and (not np.isfinite(encoder_ratio_cap) or not 0 < encoder_ratio_cap <= 1):
        raise ValueError('Encoder ratio cap must be finite and in (0, 1]')
    if background_gain is not None:
        if (objective != 'ctc' or encoder_ratio_cap is not None
                or not np.isfinite(background_gain) or not 0 <= background_gain <= 1):
            raise ValueError('Background CTC needs a finite gain in [0,1], CTC and no encoder cap')
        from background_ctc import mask as background_mask, empty_target_loss
        background_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        background_examples = tf.Variable(0, dtype=tf.int64, trainable=False)
    if positive_gain is not None:
        if (objective != 'ctc' or background_gain is None or targets.shape[-1] != 6
                or encoder_ratio_cap is not None or not np.isfinite(positive_gain)
                or not 0 <= positive_gain <= 1):
            raise ValueError('Positive CTC needs compact CTC, background markers and a finite gain in [0,1]')
        from positive_ctc import mask as positive_mask, keyword_loss
        positive_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        positive_examples = tf.Variable(0, dtype=tf.int64, trainable=False)
    if intent_gain is not None:
        if (objective != 'ctc' or targets.shape[-1] != 6 or encoder_ratio_cap is not None
                or not np.isfinite(intent_gain) or not 0 <= intent_gain <= 1 or gain <= 0):
            raise ValueError('Intent CTC needs compact targets, positive auxiliary gain and finite intent gain in [0,1]')
        from conditional_ctc import compact_loss
        intent_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        intent_examples = tf.Variable(0, dtype=tf.int64, trainable=False)
    temporal = tf.keras.layers.Reshape((STEPS, 60), name='training_phone_sequence')(
        model.get_layer('classifier').input)
    logits = tf.keras.layers.Dense(targets.shape[-1], name='training_phone_logits',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=919))(temporal)
    teacher = tf.constant(targets)
    full = tf.keras.Model(model.input, (model.output, logits))
    bounded = encoder_ratio_cap is not None and gain > 0
    if bounded:
        classifier = {id(v) for v in model.get_layer('classifier').trainable_variables}
        binary_variables = {id(v) for v in model.trainable_variables}
        encoder_indices = frozenset(i for i, v in enumerate(full.trainable_variables)
                                    if id(v) in binary_variables-classifier)
        balance_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        balanced_steps = tf.Variable(0, dtype=tf.int64, trainable=False)
        scale_sum = tf.Variable(0., trainable=False)
        scale_min = tf.Variable(1., trainable=False)
        maximum_ratio = tf.Variable(0., trainable=False)

    def phone_loss(q, phones):
        if objective == 'ctc':
            labels = tf.argmax(q, axis=-1, output_type=tf.int32)
            lengths = tf.math.count_nonzero(labels, axis=-1, dtype=tf.int32)
            losses = tf.nn.ctc_loss(labels, phones, lengths,
                tf.fill((tf.shape(phones)[0],), tf.shape(phones)[1]),
                logits_time_major=False, blank_index=0)
            return tf.reduce_mean(losses/tf.cast(tf.maximum(lengths, 1), losses.dtype))
        if objective == 'conditional':
            # CTC blank is an alignment marker, not an acoustic silence label.
            # Fit the phone identity separately; uncertain/blank frames carry
            # only their actual nonblank posterior mass as supervision weight.
            q = q[..., 1:]
            weight = tf.reduce_sum(q, axis=-1)
            q = tf.math.divide_no_nan(q, weight[..., None])
            logp = tf.nn.log_softmax(phones[..., 1:], axis=-1)
        else:
            logp = tf.nn.log_softmax(phones, axis=-1)
            weight = .05+.95*(1-q[..., 0])
        divergence = tf.reduce_sum(q*(tf.math.log(tf.maximum(q, 1e-30))-logp), axis=-1)
        numerator, denominator = tf.reduce_sum(weight*divergence), tf.reduce_sum(weight)
        return (tf.math.divide_no_nan(numerator, denominator) if objective == 'conditional'
                else numerator/denominator)

    @tf.function
    def train(x, y, weights):
        with tf.GradientTape(persistent=bounded) as tape:
            probability, phones = full(x, training=True)
            binary = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], probability)*weights)
            ids = tf.cast(y[:, 1], tf.int32)
            present = ids > 0
            q = tf.gather(teacher, tf.boolean_mask(ids, present)-1)
            phonetic = phone_loss(q, tf.boolean_mask(phones, present))
            if intent_gain:
                phonetic += intent_gain*compact_loss(q, tf.boolean_mask(phones, present))
            if background_gain is not None:
                ambient = background_mask(y[:, :3] if positive_gain is not None else y)
                if background_gain:
                    phonetic += background_gain*empty_target_loss(tf.boolean_mask(phones, ambient))
            if positive_gain is not None:
                ordinary_positive = positive_mask(y)
                if positive_gain:
                    phonetic += positive_gain*keyword_loss(tf.boolean_mask(phones, ordinary_positive))
            loss = binary+gain*phonetic
        variables = full.trainable_variables
        if bounded:
            gradients = []
            for task in (binary, phonetic):
                values = tape.gradient(task, variables)
                gradients.append([g if g is not None else tf.zeros_like(v)
                                  for g, v in zip(values, variables, strict=True)])
            combined, scale, ratio = balance_encoder_gradients(
                *gradients, encoder_indices, gain, encoder_ratio_cap)
            balance_steps.assign_add(1)
            balanced_steps.assign_add(tf.cast(scale < 1, tf.int64))
            scale_sum.assign_add(scale)
            scale_min.assign(tf.minimum(scale_min, scale))
            maximum_ratio.assign(tf.maximum(maximum_ratio, ratio))
        else:
            combined = tape.gradient(loss, variables)
        model.optimizer.apply_gradients(zip(combined, variables))
        if background_gain is not None:
            background_steps.assign_add(1)
            background_examples.assign_add(tf.math.count_nonzero(ambient))
        if positive_gain is not None:
            positive_steps.assign_add(1)
            positive_examples.assign_add(tf.math.count_nonzero(ordinary_positive))
        if intent_gain is not None:
            intent_steps.assign_add(1)
            intent_examples.assign_add(tf.math.count_nonzero(present))
        return loss

    def intent_metadata():
        if intent_gain is None:
            return None
        return {'gain': intent_gain, 'steps': int(intent_steps), 'paired_examples': int(intent_examples),
                'classes': 4, 'warmup_applied': False,
                'objective': 'Paired original-intent cross-entropy over raw CTC likelihoods; no score/length correction. Added inside the existing auxiliary gain; omitted from frozen-head warmup and scalar export.'}

    def positive_metadata():
        if positive_gain is None:
            return None
        return {'gain': positive_gain, 'steps': int(positive_steps),
                'examples': int(positive_examples), 'target': [1, 2],
                'target_meaning': 'Compact synthesis-intent [hey,nyx], not a transcript of continuation speech.',
                'normalization': 'Mean eligible-example CTC NLL divided by two target tokens; zero for an empty cohort.'}

    def background_metadata():
        if background_gain is None:
            return None
        return {'gain': background_gain, 'steps': int(background_steps),
                'examples': int(background_examples), 'frames_per_example': STEPS,
                'target': 'Empty compact sequence for explicitly marked existing ambient negatives; mean all-blank NLL per frame. Marker follows the batch shuffle; positive and paired rows are rejected.'}

    def balance_metadata():
        return {'enabled': bounded, 'steps': int(balance_steps) if bounded else 0,
                'balanced_steps': int(balanced_steps) if bounded else 0,
                'mean_scale': float(scale_sum/tf.cast(balance_steps, tf.float32)) if bounded and int(balance_steps) else None,
                'minimum_scale': float(scale_min) if bounded else None,
                'maximum_scaled_encoder_ratio': float(maximum_ratio) if bounded else None,
                'scope': 'Encoder auxiliary gradient capped before shared global clipping and Adam; auxiliary-head gradient and binary classifier gradient retain their original definitions. Logged loss remains BCE+gain*phone_loss.'}

    def audit(features):
        if objective == 'ctc':
            from phone_sequences import collapse, edit_distance
            total_loss = edits = phones = exact = 0
            for start in range(0, len(features), 32):
                x = np.clip(features[start:start+32].astype(np.float32)/25.6, 0, 26)
                q = targets[start:start+32]
                logits = full(x, training=False)[1]
                total_loss += float(phone_loss(tf.constant(q), logits))*len(x)
                for reference, prediction in zip(q.argmax(-1), np.asarray(logits).argmax(-1), strict=True):
                    reference = reference[reference != 0].tolist()
                    prediction = collapse(prediction)
                    edits += edit_distance(reference, prediction)
                    phones += len(reference)
                    exact += reference == prediction
            return {'ctc_loss': total_loss/len(features), 'sequences': len(features),
                    'exact_sequences': exact, 'target_phones': phones, 'phone_edits': edits,
                    'phone_error_rate': edits/phones}
        total_weight = total_kl = 0.
        nonblank_total = nonblank_correct = blank_total = blank_correct = 0
        for start in range(0, len(features), 32):
            x = np.clip(features[start:start+32].astype(np.float32)/25.6, 0, 26)
            q = targets[start:start+32]
            logits = np.asarray(full(x, training=False)[1])
            labels = q.argmax(-1)
            if objective == 'conditional':
                q = q[..., 1:]
                weight = q.sum(-1)
                q = np.divide(q, weight[..., None], out=np.zeros_like(q), where=weight[..., None] > 0)
                logits = logits[..., 1:]
                prediction = 1+logits.argmax(-1)
            else:
                weight = .05+.95*(1-q[..., 0])
                prediction = logits.argmax(-1)
            logp = np.asarray(tf.nn.log_softmax(logits, axis=-1))
            divergence = np.sum(q*(np.log(np.maximum(q, 1e-30))-logp), axis=-1)
            total_kl += float(np.sum(weight*divergence))
            total_weight += float(weight.sum())
            nonblank = labels != 0
            nonblank_total += int(nonblank.sum())
            nonblank_correct += int(((prediction == labels) & nonblank).sum())
            blank_total += int((~nonblank).sum())
            blank_correct += int(((prediction == labels) & ~nonblank).sum())
        return {'weighted_kl': total_kl/total_weight if total_weight else 0.,
                'nonblank_frames': nonblank_total, 'nonblank_correct': nonblank_correct,
                'blank_frames': blank_total, 'blank_correct': blank_correct if objective == 'joint' else None}

    def warmup(features, steps, learning_rate, batch_size=32):
        """Fit only the new head to cached, frozen training encoder states."""
        if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0 or batch_size < 1:
            raise ValueError('Invalid phone-head warmup settings')
        if len(features) != len(targets) or not len(features):
            raise ValueError('Warmup features must cover the complete phone bank')

        def digest(variables):
            result = hashlib.sha256()
            for variable in variables:
                result.update(variable.numpy().tobytes())
            return result.hexdigest()

        before = digest(model.weights)
        optimizer_before = digest(model.optimizer.variables)
        initial_fit = audit(features)
        encoder = tf.keras.Model(model.input, temporal)
        cache = np.concatenate([
            np.asarray(encoder(np.clip(features[start:start+32].astype(np.float32)/25.6, 0, 26),
                               training=False))
            for start in range(0, len(features), 32)
        ])
        head = full.get_layer('training_phone_logits')
        optimizer = tf.keras.optimizers.Adam(learning_rate, global_clipnorm=5)

        @tf.function
        def update(z, q):
            with tf.GradientTape() as tape:
                loss = phone_loss(q, head(z))
            variables = head.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))

        rng = np.random.default_rng(1223)
        schedule = hashlib.sha256()
        history = []
        for step in range(1, steps+1):
            ids = rng.integers(0, len(features), size=batch_size, dtype=np.int64)
            schedule.update(ids.astype('<i8', copy=False).tobytes())
            update(cache[ids], targets[ids])
            if step % 250 == 0 or step == steps:
                fit = audit(features)
                history.append({'step': step, 'fit': fit,
                                'head_weights_sha256': digest(head.weights),
                                'schedule_sha256': schedule.hexdigest()})
                print(json.dumps({'phone_warmup_step': step, **fit}), flush=True)
        if before != digest(model.weights) or optimizer_before != digest(model.optimizer.variables):
            raise ValueError('Phone-head warmup changed the binary model or its optimizer')
        return {'steps': steps, 'learning_rate': learning_rate, 'batch_size': batch_size,
                'objective': objective,
                'seed': 1223, 'schedule_sha256': schedule.hexdigest(),
                'head_weights_sha256': digest(head.weights),
                'encoder_cache_sha256': hashlib.sha256(cache.astype('<f4', copy=False).tobytes()).hexdigest(),
                'binary_weights_before_sha256': before, 'binary_weights_after_sha256': digest(model.weights),
                'binary_optimizer_before_sha256': optimizer_before,
                'binary_optimizer_after_sha256': digest(model.optimizer.variables),
                'initial_fit': initial_fit, 'history': history,
                'scope': 'Training-bank fit only; separate optimizer and cached inference-mode encoder states.'}

    train.audit = audit
    train.phone_loss = phone_loss
    train.warmup = warmup
    train.auxiliary_model = full
    train.balance_metadata = balance_metadata
    train.background_metadata = background_metadata
    train.positive_metadata = positive_metadata
    train.intent_metadata = intent_metadata
    return train
