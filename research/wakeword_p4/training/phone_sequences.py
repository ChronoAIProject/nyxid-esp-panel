"""Training-only CTC weak labels; blank zero is an alignment marker."""
import numpy as np


def collapse(tokens):
    return [int(token) for i, token in enumerate(tokens)
            if token != 0 and (i == 0 or token != tokens[i-1])]


def pack_teacher_targets(probabilities, steps):
    q = np.asarray(probabilities)
    if (q.ndim != 3 or not len(q) or q.shape[-1] < 2 or steps < 1
            or not np.isfinite(q).all() or np.any(q < 0)
            or not np.allclose(q.sum(-1), 1, atol=1e-5)):
        raise ValueError('Invalid full-resolution teacher probabilities')
    sequences = [collapse(row.argmax(-1)) for row in q]
    packed = np.zeros((len(q), steps), np.int32)
    for i, sequence in enumerate(sequences):
        # Equal adjacent labels require an intervening blank in a valid path.
        minimum = len(sequence)+sum(a == b for a, b in zip(sequence, sequence[1:]))
        if not sequence or minimum > steps:
            raise ValueError('Empty or temporally impossible teacher CTC sequence')
        packed[i, :len(sequence)] = sequence
    return np.eye(q.shape[-1], dtype=np.float32)[packed], sequences


def edit_distance(reference, prediction):
    previous = list(range(len(prediction)+1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]
