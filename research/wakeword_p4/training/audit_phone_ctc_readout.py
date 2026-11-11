"""Compare greedy/beam CTC readouts on the fixed training bank; no model update."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from phone_distillation import Sampler, supervisor
from phone_sequences import collapse, edit_distance


def beam_sequences(logits, width=32):
    import tensorflow as tf
    # TensorFlow's beam decoder requires blank last; our training blank is 0.
    values = tf.convert_to_tensor(logits, dtype=tf.float32)
    reordered = tf.concat((values[..., 1:], values[..., :1]), axis=-1)
    result, _ = tf.nn.ctc_beam_search_decoder(tf.transpose(reordered, (1, 0, 2)),
        tf.fill((tf.shape(values)[0],), tf.shape(values)[1]), beam_width=width, top_paths=1)
    sequences = [[] for _ in range(int(values.shape[0]))]
    for index, label in zip(result[0].indices.numpy(), result[0].values.numpy(), strict=True):
        sequences[int(index[0])].append(int(label)+1)
    return sequences


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'features', 'phone-bank', 'checkpoint', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh result path')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(a.upstream.resolve()))
    from transfer import build_model
    index = (a.features/'features.json').read_bytes()
    bank = Sampler(a.phone_bank, json.loads(index), hashlib.sha256(index).hexdigest(), 335, 'ctc')
    model, _ = build_model(335)
    train = supervisor(model, bank.targets, .01, 'ctc')
    train.auxiliary_model.load_weights(a.checkpoint)
    before = [v.numpy().copy() for v in train.auxiliary_model.weights]
    totals = {name: {'edits': 0, 'decoded_phones': 0, 'exact_sequences': 0} for name in ('greedy', 'beam32')}
    rows = []
    for start in range(0, len(bank.features), 16):
        x = np.clip(bank.features[start:start+16].astype(np.float32)/25.6, 0, 26)
        logits = np.asarray(train.auxiliary_model(x, training=False)[1])
        predictions = {'greedy': [collapse(row) for row in logits.argmax(-1)],
                       'beam32': beam_sequences(logits)}
        labels = bank.targets[start:start+16].argmax(-1)
        for i, target in enumerate(labels):
            reference = target[target != 0].tolist()
            row = {'row': start+i, 'reference': reference}
            for name, sequences in predictions.items():
                sequence = sequences[i]
                error = edit_distance(reference, sequence)
                totals[name]['edits'] += error
                totals[name]['decoded_phones'] += len(sequence)
                totals[name]['exact_sequences'] += reference == sequence
                row[name] = {'sequence': sequence, 'edits': error}
            rows.append(row)
    for old, new in zip(before, train.auxiliary_model.weights, strict=True):
        np.testing.assert_array_equal(old, new.numpy())
    phones = sum(len(row['reference']) for row in rows)
    for name, values in totals.items():
        values['phone_error_rate'] = values['edits']/phones
    report = {'scope': __doc__, 'source_sha256': sha(Path(__file__)),
              'bank_sha256': bank.bank_sha256, 'sequence': bank.sequence_metadata,
              'checkpoint_sha256': sha(a.checkpoint), 'weights_unchanged': True,
              'beam_width': 32, 'sequences': len(rows), 'target_phones': phones,
              'totals': totals, 'rows': rows,
              'limits': 'Training-only weak teacher labels. No fit gate is changed, no candidate promoted, and no beam decoder is added to runtime. Neither readout proves wake recognition, phonetic truth or acoustic parity.'}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
