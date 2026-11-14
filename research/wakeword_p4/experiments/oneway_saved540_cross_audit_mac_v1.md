# Mac independent read of the original one-way 540-case packet

This is a read-only audit of previously consumed numerical evidence, not a new
recognition experiment. It loads NumPy and original NPZ/JSON bytes; no model,
audio, frontend, training or inference library is used. All startup outputs and
all four arms × the original 135 requests remain included.

The reader binds the original 8,546,115-byte archive, its committed manifest at
`57f7adf`, all 553 payloads, all 13 original committed receipts, and the original
public source/receipt locks. The formal-weight → compiled-export → screen
metadata chain is checked for all four arms. Actual H5/TFLite/audio/training
assets are absent, so those bytes and their original computation are **not**
independently verified by this packet read.

For every saved case it independently checks the Cartesian product, NPZ dtypes,
complete feature/output lengths, input quantization bytes, output repeats,
full clocks, phase errors, raw and five-step errors and unchanged numerical
gates. Same-input raw feature bytes must match across all four models. Float
reference output is stored as float64, repeat as float32: compare their bytes
after the original lossless float32-to-float64 promotion, not across dtypes.

It recomputes the entire GPU v2 summary and the complete matched reduction at
`0acd924`, including all pass/fail transitions, signed metric differences and
every failed case's post-startup error windows. A second five-window arithmetic
form checks convolution to a fixed 1e-15 roundoff bound. The original v1 KeyError
and stopped continuation receipts remain authenticated, never replaced.

The initial Mac reader at `a95dea4` stopped on case11: the five-window maximum
was `0.04736376297660172` versus original `0.04736376297660171`. It incorrectly
required cross-machine reduction scalars to be bit-identical. Its original
exit1/log/source revision are retained. This is not an inference mismatch.
The revised read uses the binary64 forward-error bound `2*gamma(n+9)` for
bounded reductions (`u=eps/2`), independently of observed errors. Every original
model gate is unchanged and must be farther from its decision boundary than
this bound. Case decisions, array identities, phase errors, clocks and repeat
bytes remain exact. Reported scalar statistics are **not** claimed byte-exact
across CPU reduction implementations. The matched scalar arithmetic is replayed
from original statistics after checking those against the saved arrays.

```sh
PYTHON experiments/audit_oneway_saved540_mac_v1.py \
  --repository-root REPO \
  --archive ORIGINAL_PACKET.tar.gz \
  --report NEW_REPORT.json
```

Use NumPy 2.4.6, matching the original arithmetic. The output refuses overwrite.
No playback, microphone, USB or production-board operation is involved. This
audit cannot establish quality, FAR, fresh-development admission, P4 resources,
or a candidate replacement. All original numerical rejections remain valid.

## Actual completed read

Reader revision `32cbd21` completed with exit0 on Mac, NumPy2.4.6/Python3.12.13.
It authenticated all553 payloads and41 original public source/receipt locks,
then checked all540 cases /96,168 saved output steps /135 unique raw inputs.
The report is `results/mac-oneway-saved540-cross-audit-v2.json`; original run
stdout is retained next to it. No model inference was performed.

| Original condition | Willow passed /135 | Nyx passed /135 |
| --- | ---: | ---: |
| two-way consistency | 125 | 126 |
| one-way consistency | 126 | 127 |

All540 decisions, failed-request lists, matched transitions and post-startup
failure-window descriptions agree with GPU's original results. Willow has
7 fail→pass and6 pass→fail; Nyx has4 and3. Neither condition qualifies all270.
All36 failed cases have five-window errors above.1 after startup; dropping
startup would not resolve these failures. These are numerical-conversion
statistics, not wake-word recognition counts or statistical efficacy evidence.

Maximum cross-machine scalar difference is `5.551115123125783e-17`, below the
derived bound (largest case bound `5.373479439185902e-14`). Minimum distance
from a mean/p99/max gate is `0.0044651171315927`, so no gate decision depends on
this rounding. Full phase errors, clocks, quantized inputs and output repeats
retain the exact checks described above. Scalar maxima are reported with
their portable arithmetic limit; no cross-machine scalar byte equality claim.
