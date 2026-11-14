# Exhaustive post-head probability gap

Plan `32f21d6` was published before execution. This is a diagnostic of the four
already rejected consistency-family final10000 models; it changes no model or
numerical gate. It extracts their original stateless MUL/ADD/LOGISTIC/QUANTIZE
operations as a derived fragment, preserving every original buffer, tensor,
quantization grid and opcode version. It invokes no full model or encoder.

For each model, two fresh builtin-reference LiteRT fragment interpreters process
all 256 int8 FC codes and the original complete witness FC sequence. All four
intermediate outputs and repeats are saved. The 792 witness final UINT8 outputs
must match the original whole-model observations exactly. The independent audit
parses both FlatBuffers without an interpreter, authenticates every original
buffer and operator/tensor definition, validates saved repeats and all witness
stage lookups, and recomputes the probability/error decomposition.

Actual result: 3632 fragment invocations completed; all repeats, witness outputs,
serialization and independent saved-array checks passed. All four enumerated
post-head UINT8 tables are identical. The largest gap to sigmoid of the nominal
float32 bounded coordinate is `0.008804371898468277` over every int8 FC code.

On the original Willow14 control/treatment witnesses, five-step p99 errors are
`.189241327083786` / `.14260083398048298`; the nominal-FC-versus-float components
remain `.19026900960191148` / `.13937517551383402`. Even subtracting the uniform
post-head bound gives lower bounds `.18043695518531772` / `.1337964620820147`,
both above the unchanged `.1` p99 gate. Thus post-head quantization alone cannot
explain these two failures. This does not identify a causal encoder layer, prove
any training intervention effective, or explain all 35 original failed cases.

The nominal target's final-probability approximation has a measured bound here;
it is not full quantized-head equality. Enumeration is native LiteRT fragment
execution, not independent exhaustive TFLM/P4 execution, recognition/FAR testing,
fresh development, acceptance, or a qualified model.

Run from the repository root in the authenticated native Python environment:

```sh
python research/wakeword_p4/experiments/diagnose_consistency_post_head_gap_mac_v1.py run \
  --payload-root "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --native-root "$PRIVATE/consistency-fixed-grid-native-four-witness-replay-v1" \
  --output-dir "$PRIVATE/consistency-post-head-gap-mac-v1"

python research/wakeword_p4/experiments/audit_consistency_post_head_gap_mac_v1.py \
  --root "$PRIVATE/consistency-post-head-gap-mac-v1" \
  --payload-root "$PRIVATE/gpu-consistency-fixed-grid-mac-witnesses-v1-authenticated/replay-payload" \
  --native-root "$PRIVATE/consistency-fixed-grid-native-four-witness-replay-v1" \
  --report "$PRIVATE/mac-consistency-post-head-gap-independent-audit-v1.json"
```

`PRIVATE` is the existing local research output root. Output paths must be new;
do not overwrite originals. The public artifact receipt records all NPZs,
derived fragments, source/plan/report identities and anonymous redownload hashes.
