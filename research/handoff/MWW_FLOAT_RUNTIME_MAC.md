# Mac float-runtime preparation

Executed on 2026-11-12 while the original v47 training continues. This uses
locally seeded random weights and three synthetic uint16 feature streams. It
does not run v46/v47, score training or development samples, choose a loss, or
establish recognition quality.

[Probe](probe_mww_float_runtime_mac.py) and
[executed report](../wakeword_p4/results/mww-float-runtime-mac-v1.json).

```sh
/Users/auric/tmp/output/wakeword-macos-research/.venv/bin/python -B \
  research/handoff/probe_mww_float_runtime_mac.py \
  --upstream /Users/auric/tmp/output/wakeword-macos-research/microWakeWord \
  --output /tmp/mww-float-runtime-fresh.json
```

The probe authenticates all tracked Python sources in the local upstream
checkout against revision `4665173cd35f1cff9a61e06fc427f124766c488e`, and verifies
that imported microWakeWord modules come from that checkout. Architecture
arguments come directly from the original v46 training record at `65ebbcc`.
This authenticates the local graph source, not the GPU's historical upstream
working-tree bytes; the latter still need an original execution receipt.

The 22,801-parameter teacher60/context150 graph builds in the existing Mac
environment: Python3.12.13, TensorFlow2.21.0, Keras3.15.1, NumPy2.5.4,
h5py3.14.0. Random H5 save/load is exact. Six streaming state buffers reset
exactly, including when a different stream is interposed between replays.
All nonstate weights and the reference window-model weights remain unchanged.
Tail lengths of zero, one and two frames follow the existing stride-three
contract.

Using the existing, fixed phase1 alignment, the streaming and full-window
representations match across all 353 mature synthetic windows. Maximum
absolute error is `1.0244548320770264e-8` for the representation and
`5.960464477539063e-8` for probability. Phase0 sensitivity is reported without
searching or changing the phase. Initial stream outputs are retained in reset
checks; only the full-window equivalence comparison requires complete context.

The actual Dense input is a **flattened 5-by-60, 300-dimensional** representation.
A future penultimate-representation diagnostic must state whether it uses this
complete classifier input or a separately defined temporal reduction. Calling
either an interchangeable 60-dimensional embedding would hide a change in the
measurement. No such reduction or objective has been selected here.

Original v46 weights, original arrays, and the exported SavedModel have not
been loaded in this preparation. Random H5 compatibility cannot substitute for
loading and authenticating the original artifact. The full actual-float
diagnostic remains conditional on original v47's complete result and the
requested exact-artifact handoff through Issue13. There is no new training,
INT8/P4 test, audio playback/capture, threshold change or candidate acceptance.
