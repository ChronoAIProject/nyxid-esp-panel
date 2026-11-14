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
