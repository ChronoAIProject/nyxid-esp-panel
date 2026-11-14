# Trained internal-state FP32: actual host MicroInterpreter

This advances the official Hey Willow → same-recipe Willow → Hey Nyx goal by
checking all four original trained exports in the real installed TFLM runtime.
It does not establish recognition quality or authorize deployment.

The prior random-model inspector and eleven-op runner remain byte-identical.
GPU's original trained exports additionally use one EXPAND_DIMS: scalar int32
axis 2, audio float32 `[1,3,40]` → `[1,3,1,40]`. The new v2 inspector supports
that exact shape operation and retains the six zero internal states, raw Dense
sigmoid, finite FP32 constants and single-input/single-output contract. The
new native runner adds installed `kernels/expand_dims.cc` and its registration,
checks all registrations, input/output counts and exact shapes. No received
model or reference is changed or reconverted to fit the older runner.

The first static intake at source `6218230` rejected the original trained input's
`shape_signature=[-1,3,40]`, before extraction or inference. The original failure
is retained. All four actual models have this symbolic batch metadata only on
the audio input and ExpandDims output; their concrete shapes are fixed batch 1.
Installed `micro_allocator.cc` lines 260–270, 357–362 and 464–486 allocate concrete
`shape()` dimensions, not `shape_signature()`. The revised inspector explicitly
allows only those two symbolic batch metadata tensors with concrete batch 1,
rejecting dynamic concrete dimensions, symbolic time/frequency/state/output
dimensions and inconsistent static signatures. The runner never resizes inputs.
This is a metadata-compatibility correction before inference; no probability
gate, model, reference or scoring rule changes.

Before inference:

1. Publish this scope and all source files. Authenticate the 527033-byte original
   GPU archive SHA `06c53a9334a59ff07ce21e85afad9e9bf29fb79da2fe4879be3e560b0b06260a`,
   exact nineteen regular members/all eighteen payloads, committed manifest,
   source/execution/result receipts, four model/H5 bindings and saved references.
   Extract only after every identity and structural check passes.
2. Freeze/publish actual compiler, full installed runtime tree, 57 translation
   units, all helper/probe sources, four unchanged models/references, intake
   result and original input bytes. Plan regeneration must match before run.
3. Run four models × four original synthetic streams × two fresh interpreters
   × 256 steps = 8192 actual MicroInterpreter invocations, retaining all startup.
   No audio, old135, fresh development or acceptance is consumed in this stage.

Probability max absolute error against both original GPU TF and builtin-reference
LiteRT must be <=1e-4 at every step/repeat; fresh-repeat output bytes must match.
All outputs, stdout/stderr, compile and failure receipts are retained. A separate
saved-output audit must recompute these claims before publication to GPU.

Original ESP_NN=0 float kernels/compiler flags and chrono-only timer adapter are
retained. Arenas are fresh and value-initialized, not poisoned-memory tests.
Host64-bit allocation is not P4 allocation/timing evidence. Original INT8
rejections and fullstream/phase/recognition/FAR/P4 gates remain in place.
