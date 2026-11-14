# Bounded head runtime preparation

The equalized exports recorded in GPU `dbcdf6e` contain a fused RELU6 FC,
followed by MUL, ADD, LOGISTIC and QUANTIZE. The complete-input research runtime
now registers MUL and ADD and reserves sixteen resolver slots. Legacy branches
keep their original fourteen registrations. The host shim enforces resolver
capacity, so a missing capacity increase fails actual initialization flow.

The complete/legacy flow review passed under undefined/bounds sanitizers, with
the original continuous output prefix unchanged. All three runtime variants
compiled against the existing real ESP32-P4 toolchain and headers with zero
warnings. Reports are `mac-bounded-head-runtime-{host-flow,p4-compile}-v1.json`.
These checks do not allocate a learned model or execute operators on P4.

## FC kernel identity

`replay_p4_fc_kernel_mac_v1.py` compiles the installed ESP-NN portable ANSI FC C
source into a Mac library and executes it on the two original saved embeddings.
The P4 header selects that ANSI function. Quantization preparation is separately
derived from the installed TFLM scalar helper; it is not an executed TFLM Prepare.
The downstream-only LiteRT fragment reproduces every original UINT8 output when
fed the original FC tensor, before testing controlled ANSI FC outputs.

Across Willow14's 167 steps, ANSI FC differs from LiteRT BUILTIN_REF at two
steps by one code; the resulting host downstream UINT8 differs at one step by
one code. Nyx48394's 229 steps are exact in this comparison. Both C replays are
repeatable. The report is `mac-p4-ansi-fc-kernel-replay-v1.json`.

This is an isolated kernel comparison with host embeddings and postops. It does
not establish complete P4 inference, optimized convolution identity, a learned
P4 resource profile, recognition quality or permission to weaken exact selftest.
The separate embedding experiment's custom FC registration is not used by this
teacher60 runtime. Its historical single-rounding comment is not evidence of
the current LiteRT scalar FC overload, which uses double scaling and round-away.

## Opt-in scalar compatibility

The complete-input branch now registers `complete_fc.h`; the legacy branch and
dependency sources retain their standard registrations. The adapter retains
the installed standard preparation of activation bounds and zero-points, then
uses a single-output int8 FC with the actual float32 scales widened separately
to double. Accumulation is int64 with a conservative int32 bound checked during
preparation; final scaling uses signed round-away and the prepared activation
clamp. Unsupported types, multiple outputs, per-channel weights and non-scalar
bias are rejected. This is an explicit host-reference compatibility choice for
this research profile, not a claim that all ESP-NN kernels match LiteRT.

`replay_complete_fc_math_mac_v1.py` compiles the actual arithmetic header as a
native Mac library. Both original saved witnesses, 396 steps in total, match
their original actual LiteRT FC and final UINT8 outputs exactly. One batched
call, individual calls and repeated calls agree. Another 4,783 constructed
codes cover signed half ties, bias/no bias, zero-points, activation clipping,
multiple depths and legal int32 accumulator extremes against an independent
exact-rational oracle. See `mac-complete-fc-math-replay-v1.json`.

`replay_complete_fc_registration_mac_v1.py` additionally compiles and executes
the real adapter Init/Prepare/Invoke and fourteen installed TFLM source files,
including the real standard FC Prepare. Its small tensor/allocator host context
and unused zero timer are test support; it does not instantiate a complete
MicroInterpreter or P4 arena. `ESP_NN=0` changes the replaced standard Eval,
not the executed standard Prepare. All 396 FC outputs again match, including
the prepared double scales and repeated receipts. Eight lifecycle controls
cover optional/present bias and rejection of unsupported output type, output
count, bias shape, per-channel quantization, overflow risk and changed Invoke
shape. The successful final report is
`mac-complete-fc-registration-replay-v2.json`.

The initial lifecycle run actually failed its wrong-output-type control: calling
the standard Prepare before validation returned early with four temporary
tensor references outstanding. That original report remains
`mac-complete-fc-registration-replay-failed-v1.json`; the exact failed header is
`history/complete_fc_failed_type_guard_v1.h.txt` (SHA matches the report).
The adapter now checks supported metadata and releases its temporary tensors
before calling standard Prepare. All eight final controls pass with zero
outstanding temporary references; no dependency source was edited.

The final production headers also compile in all three real P4 translation-unit
variants with zero warnings (`mac-complete-fc-adapter-p4-compile-v2.json`). The
actual runtime host-flow check still preserves the legacy continuous prefix and
passes eight selftests under undefined/bounds sanitizers
(`mac-complete-fc-adapter-host-flow-v1.json`). These remain isolated arithmetic,
host lifecycle and compilation checks. Full-model P4 inference, other kernels,
resource use and recognition quality are still unverified for a new candidate;
the exact NYXKWS02 selftest and model admission requirements remain in force.

Reproduce both native replays using the diagnostic Python with NumPy/LiteRT,
the original authenticated witness payload and Mac integer replay, and a new
output directory (existing evidence is never overwritten):

```sh
python research/wakeword_p4/experiments/replay_complete_fc_math_mac_v1.py \
  --gpu-ref fd2d227 --payload-root /path/to/payload \
  --integer-replay-root /path/to/mac-replay-v1 \
  --ansi-replay-root /path/to/p4-ansi-fc-host-replay-v1 --output-dir /new/math-replay
python research/wakeword_p4/experiments/replay_complete_fc_registration_mac_v1.py \
  --gpu-ref 5eaac0d --payload-root /path/to/payload \
  --integer-replay-root /path/to/mac-replay-v1 --output-dir /new/registered-replay
```

The saved arrays, original failed lifecycle report, final reports and bound
source headers are published as
[`mac-complete-fc-native-replays-v1.tar.gz`](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/mac-complete-fc-native-replays-v1.tar.gz)
(15,945 bytes, SHA256
`a24f3206439570de88db79b03a5bdbe15ceed49753d6e95453bf1a040e11e5b1`).
Anonymous redownload verified the whole archive and all nineteen manifest
payload members. The separate initial host harness build failure is retained
in `mac-complete-fc-initial-host-build-failure-v1.json`; the harness subsequently
adopted TFLM's required virtual-delete macro and compiled successfully.

## Candidate handoff

The original and equalized candidates remain rejected. The P4 profile preparer
currently authenticates the original fixed120000 training/model identity.
A later qualified fine-tuned export needs an explicit derivative receipt/profile
extension linking the immutable original training, fine-tuning result and new
export; changing the original report's model hash is invalid. Numerical gates,
non-null fresh-development selection and a new exact model-bound NYXKWS02
fixture remain required before a learned model is installed on the test board.
