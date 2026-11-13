# Independent context225 temporal and crop-support review

Mac authenticates the exact MixedNet, Stream, modes and converter source bytes
declared in GPU's frozen `65c3a23` v3 probe plan. No TensorFlow, learned weights,
media or detector inference is used by this review.

| Nominal input frames | First valid conv outputs | Outputs after four depthwise blocks | Effective input frames | Unused final frames |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 49 | 5 | 149 | 1 |
| 225 | 74 | 30 | 224 | 1 |
| 300 | 99 | 55 | 299 | 1 |

The first kernel is5 with stride3. The subsequent valid depthwise kernels
5/9/13/21 reduce the temporal output by4+8+12+20=44. The flatten head retains
all remaining temporal slots. For context225, the receptive span is therefore
5+3×(44+29)=224 frames. The same one-frame remainder exists at context150;
this is not evidence that the older two-voice pilot failed because context225
changed the temporal phase.

At the end E of a streaming input step, the effective input is `[E-224, E)`.
The nominal offline225 window `[E-225, E)` only consumes `[E-225, E-1)`.
Thus the same-time diagnostic offline reference is the latest224 actual frames
followed by one unused dummy frame. This is determined by indexing, not by
searching offsets or relaxing numerical tolerance. The dummy is not sent to
the detector or added to the original waveform. Actual streaming sample clocks
must remain unchanged for latency/FAR scoring.

The independent checker builds ordered dependency hashes for every valid
temporal operation and explicitly simulates the streaming ring buffers. All
768 compared clocks across the three contexts match that derived reference;
all nominal windows differ. Altering only the unused input has no effect.
These symbolic identities establish dependency alignment, not equality of
learned floating-point or quantized outputs.

The actual model also ignores the final nominal feature when checking positive
crop support. With that accounted for, all3120000 constructed crops retain
source sample0 through the full generated prefix and RIR: 48000 leading samples,
16000 trailing samples, jitter3..27, RIR0/960, both standalone and command-tail
containers, and all positive prefix+RIR lengths up to the recorded31680 bound.
This does not verify actual feature files, sampling draws, phonetic truth or
the continuous peak allocation of the deployed model.

GPU subsequently published `ce96f31` actual initial-float results: six complete
training-only PCM streams, exact twice-reset traces, phase-aware max error
2.980232238769531e-7 at the original1e-4 tolerance. It preserves both failed
attempts and nominal-window differences. That is separate reported numeric
evidence; this Mac review has not reloaded those learned weights or arrays.
Trained INT8 export, actual batches and P4 remain later checks.

Reproduce the symbolic/source review from the repository root, using an exact
upstream checkout with the hashes recorded in the result:

```sh
python3 research/wakeword_p4/experiments/review_hey_context225_phase_v1.py \
  --upstream-root /path/to/microWakeWord/microwakeword \
  --output /tmp/mac-context225-phase-reproduced.json
```

The output path must be new. The JSON should exactly reproduce
`research/wakeword_p4/results/mac-context225-phase-and-support-v1.json`.
The standalone Muse probe and panel app are unchanged by this review.
