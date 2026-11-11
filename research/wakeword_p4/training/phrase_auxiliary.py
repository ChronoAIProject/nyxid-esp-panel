"""Training-only short-phrase auxiliary loss with complete source PCM support."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


TEXT_CLASSES = {'Hey Nyx.': 0, 'Hey Nicks.': 0, 'Hey Nix!': 0,
                'Hey Nicks?': 0, 'Hey Nick.': 1, 'Hey Next.': 2, 'Hey Mix.': 3}


class PhraseAuxiliary:
    def __init__(self, model, features, manifest, weight):
        manifest = Path(manifest)
        audit = json.loads(manifest.read_text())
        index = Path(features) / 'features.json'
        if (hashlib.sha256(index.read_bytes()).hexdigest() != audit['feature_index_sha256']
                or audit['context_frames'] != 150 or audit['final_data_accessed']):
            raise ValueError('Auxiliary manifest context, split scope or feature hash mismatch')
        rows = json.loads(index.read_text())
        lookup = {(r['engine'], r['path']): r for r in rows
                  if r['split'] == 'training' and r['variant'] == 0}
        self.groups = {}
        for record in audit['records']:
            if not record['entire_source_fits'] or record['text'] not in TEXT_CLASSES:
                continue
            row = lookup[(record['engine'], record['path'])]
            if (row['sha256'] != record['source_sha256'] or row['offset'] != record['feature_offset']
                    or row['text'] != record['text'] or row['samples'] != record['samples']):
                raise ValueError('Auxiliary manifest row mismatch')
            label = TEXT_CLASSES[row['text']]
            if row['label'] != int(label == 0):
                raise ValueError('Auxiliary target contradicts original binary label')
            # teacher60 stride3/kernel5 ignores the last input feature frame.
            # Shift crop one frame later and exclude sources that no longer fit.
            end = record['feature_crop_end'] + 1
            if (end-150)*160 > row['lead_frame']*160:
                continue
            if (not isinstance(end, int) or end < 150 or end > row['frames']
                    or (end-150)*160 > row['lead_frame']*160
                    or (end+1)*160 < row['lead_frame']*160+row['samples']):
                raise ValueError('Incomplete auxiliary crop')
            self.groups.setdefault(label, {}).setdefault(row['engine'], []).append((row, end))
        if set(self.groups) != set(range(4)):
            raise ValueError('Require all four auxiliary classes')
        self.raw = np.memmap(Path(features)/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
        self.rng = np.random.default_rng(67726)
        # Seeded initializer avoids advancing the original training initializer RNG.
        logits = tf.keras.layers.Dense(4, name='training_phrase_auxiliary',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=67726))(model.layers[-1].input)
        self.network = tf.keras.Model(model.inputs, [model.output, logits])
        self.weight = float(weight)
        self.optimizer = model.optimizer
        self.optimizer.build(self.network.trainable_variables)
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction='none')
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.report = {'classes': TEXT_CLASSES, 'weight': self.weight, 'seed': 67726,
                       'batch_size': 24, 'samples_per_class': 6,
                       'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                       'groups': {str(k): {e: len(v) for e,v in es.items()} for k,es in self.groups.items()},
                       'policy': 'Separate 24 complete-source variant0 crops, equal classes and equal engines within class; shift end+1 and exclude sources exceeding24160samples to account for teacher60 unused suffix frame; no extra binary supervision, no augmentation; original binary batch and loss unchanged; deploy binary model only',
                       'limits': 'Text intentions, not verified phone classes. No ambient class or phonetic endpoint truth. Extra forward pass/gradient changes training compute.'}

    def batch(self):
        x, y = [], []
        for label in range(4):
            engines = sorted(self.groups[label])
            for _ in range(6):
                rows = self.groups[label][str(self.rng.choice(engines))]
                row, end = rows[int(self.rng.integers(len(rows)))]
                start = row['offset'] + end - 150
                x.append(self.raw[start:start+150])
                y.append(label)
        return np.clip(np.asarray(x, dtype=np.float32)/25.6, 0, 26), np.asarray(y, dtype=np.int32)

    @tf.function(reduce_retracing=True)
    def step(self, x, y, weights, auxiliary_x, auxiliary_y):
        with tf.GradientTape() as tape:
            prediction, _ = self.network(x, training=True)
            _, logits = self.network(auxiliary_x, training=True)
            binary = tf.reduce_mean(self.bce(y, prediction) * weights)
            auxiliary = self.ce(auxiliary_y, logits)
            loss = binary + self.weight * auxiliary
        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
