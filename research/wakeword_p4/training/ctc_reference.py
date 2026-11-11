"""Small independent float64 CTC forward/backward reference for diagnostics.

This is deliberately offline NumPy code, not a training or firmware path.
"""
import numpy as np


def loss_and_gradient(logits, sequence):
    x = np.asarray(logits, np.float64)
    if (x.ndim != 2 or not len(x) or not np.isfinite(x).all()
            or not sequence or any(type(v) is not int or not 0 < v < x.shape[1] for v in sequence)):
        raise ValueError('Expected finite logits and a nonempty nonblank sequence')
    minimum = len(sequence)+sum(a == b for a, b in zip(sequence, sequence[1:]))
    if len(x) < minimum:
        raise ValueError('Temporally impossible CTC target')
    shifted = x-x.max(-1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
    states = [0]
    for token in sequence:
        states.extend((token, 0))
    forward = np.full((len(x), len(states)), -np.inf)
    backward = np.full_like(forward, -np.inf)
    forward[0, :2] = logp[0, states[:2]]
    for time in range(1, len(x)):
        for state, token in enumerate(states):
            incoming = [forward[time-1, state]]
            if state:
                incoming.append(forward[time-1, state-1])
            if state >= 2 and token != 0 and token != states[state-2]:
                incoming.append(forward[time-1, state-2])
            forward[time, state] = np.logaddexp.reduce(incoming)+logp[time, token]
    total = np.logaddexp.reduce(forward[-1, -2:])
    backward[-1, -2:] = 0.
    for time in range(len(x)-2, -1, -1):
        for state, token in enumerate(states):
            outgoing = [backward[time+1, state]+logp[time+1, token]]
            if state+1 < len(states):
                outgoing.append(backward[time+1, state+1]+logp[time+1, states[state+1]])
            if state+2 < len(states) and states[state+2] != 0 and states[state+2] != token:
                outgoing.append(backward[time+1, state+2]+logp[time+1, states[state+2]])
            backward[time, state] = np.logaddexp.reduce(outgoing)
    posterior = np.exp(forward+backward-total)
    gradient = np.exp(logp)
    for state, token in enumerate(states):
        gradient[:, token] -= posterior[:, state]
    if not np.allclose(posterior.sum(-1), 1, atol=1e-10, rtol=0):
        raise ValueError('CTC state posterior does not normalize')
    return float(-total), gradient
