# Prospective Mac original-snapshot gradient replay

Publish this plan and `replay_engine_gradients_mac_v1.py` before running learned
inference/autodiff on the handoff. Anonymous download and all 170 member hashes
have already been checked without model execution. The immutable archive SHA is
`982a9b615c08c201157180f7abe01de08c84e421fbe436766ac366fc028ed238`,
source result `6b667b4a`, handoff `f966d99b`, original producer `518c72ee`.

Use the existing native Mac CPU runtime: TensorFlow 2.21.0, Keras 3.15.1, NumPy
2.4.6 and h5py 3.14.0, two intra-op threads, deterministic operations. Authenticate
all 19 pinned upstream Python files before importing the network. Execute only
the unchanged pure architecture helper and unchanged nested reversal function;
do not execute the GPU-only factory or any original training CLI.

First load each actual same-instance initial H5. Compare every one of its 31
model arrays byte-for-byte with its original variable snapshot and end-state
inventory. Load the two original domain variables from that same snapshot.
Verify all 17 trainable array identities/order/shapes and both original indices.
This is not historical weight reconstruction or a newly randomized probe.

Execute all 50 original main/auxiliary pairs for each word, in original order,
two fresh deterministic CPU evaluations per pair. No optimizer or update. Compute
S, direct D, reversal-0 D and reversal-1 D. Require exact None patterns, zero/
negative encoder gradients, unchanged domain-head gradients, exact CPU repeat
bytes and unchanged initial/end variables. Save actual CPU logits/losses/features
and every connected gradient, bound to original case and auxiliary hashes.

Cross-platform forward/loss/gradient arrays use the prospectively fixed rule
`abs(delta) <= 1e-4 + 1e-4*abs(saved)`. This is a host diagnostic comparison,
not the deployment FP32 absolute-1e-4 admission gate. Independently compute the
saved domain bias gradient as mean(softmax(logits)-onehot) in NumPy float64;
require absolute error <=1e-6. Retain all arrays and metrics, including numeric
failures; no point, case, tolerance or coefficient will be selected from results.

Structural identity failures stop and retain the failed directory. If numeric
comparison fails, finish all 100 original cases and write the failed result;
do not silently repair weights or loosen criteria. Preserve the original handle
across observation timeouts. No audio, new corpus, reserved development/acceptance,
board write, training, selected coefficient or efficacy claim. The active goal
and all deployment/recognition/audio-lifecycle requirements remain incomplete.
