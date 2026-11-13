# Broad context225 independent batch-flow review

Reviewed GPU commit `2d7b5669512049523ad968191968cf834299a4ef` using repository bytes, without checking out or changing the GPU branch. The trainer, adapter and smoke entry point match the pushed smoke lock. The original trainer SHA256 is `2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904`; the independently transformed source SHA256 is `228c392400a95d1e89ffb0409a002fb71b5f4c2f4c581f0100a4f1d1ed8b7baa`, matching the GPU's source audit.

## What was checked

The actual adapted `scale`, `sequence`, `sample`, `prefix_sample` and `batch` functions run against identifiable, read-only, constructed feature arrays. Each row and frame has a different integer marker. An observer records the source slices actually read, and an independent assembly reconstructs the expected final batch through negative replacement and permutation. This specifically checks the earlier class of error where sampling positives succeeded but later replacement erased them before model input.

Across both target arms, training/calibration modes and exact repeats:

- 400 batches and 12,800 own-target positive windows retain their exact source slices, labels and weights in the final returned tensor.
- Each batch contains 32 unchanged positives, 32 surviving ordinary confusing negatives, 16 prefix negatives and 48 ambient negatives. Prefix replacement occupies negative slots 32–47 before permutation.
- All 6,400 prefix windows come from Piper/Kokoro, with the prescribed 15–35 frame gaps. Qwen remains available for ordinary positive and negative sampling.
- Constructed same-seed schedules match across arms by engine, conditioning/variant, own/reciprocal/common role and crop-relative clock. Training and calibration produce the same tensors under this recipe's one negative candidate and zero masking widths.
- Source arrays remain unchanged; repeating every run reproduces its tensor digest exactly.
- Deliberately substituting a negative for one returned positive is rejected. Changing that positive's label and weight is also rejected. Shape/count checks alone would not detect the first corruption.

The transformed full trainer compiles, contains no calls to `validate`, retains a save at `step == args.steps`, and invokes the final batch guard once at the batch return. Source inspection confirms the exact context225 snapshot loads after the original context150-only initialization block and before batchnorm policy application; no classifier reset follows that load. These source checks do not prove an optimizer execution completed.

## Evidence and limits

Result: `research/wakeword_p4/results/mac-hey-broad-batch-flow-review-v1.json`.

Reproduce from the repository with NumPy and the referenced GPU commit available:

```sh
python3 research/wakeword_p4/experiments/review_hey_broad_batch_flow_v1.py \
  --gpu-ref 2d7b566 \
  --output NEW_PATH.json
```

This is an independent constructed flow check of the actual pinned functions. It does not load the private broad feature pack, run TensorFlow or GPU updates, certify the real paired draw schedule, infer INT8/float equality, score recognition/FAR, open acceptance data, play audio or record a microphone. GPU's real 106,956-presentation feature and CPU-batch audits remain separate evidence. The two-step GPU smoke is only an execution/export check, not a trained candidate or verification of the formal 120,000-step learning-rate schedule.

No blocking defect was found in the reviewed flow. Continue actual smoke/formal training under the fixed common recipe; do not repeat this constructed review as an additional prerequisite. After training, the actual paired schedule, fixed-final weights and trained streaming/INT8 outputs still require their own checks. Actual stream clocks remain unchanged; the phase-alignment dummy is diagnostic only.
