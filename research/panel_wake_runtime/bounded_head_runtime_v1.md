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

## Candidate handoff

The original and equalized candidates remain rejected. The P4 profile preparer
currently authenticates the original fixed120000 training/model identity.
A later qualified fine-tuned export needs an explicit derivative receipt/profile
extension linking the immutable original training, fine-tuning result and new
export; changing the original report's model hash is invalid. Numerical gates,
non-null fresh-development selection and a new exact model-bound NYXKWS02
fixture remain required before a learned model is installed on the test board.
