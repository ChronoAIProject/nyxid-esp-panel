# Optional recovery of overwritten training draws

This prepares the next integration check, not a new formal training run. Review
the complete fixed/per_draw result before choosing another training comparison.
Do not modify executing source files, restart either arm, retune coefficients,
change PCM speed, consume new development/final audio or operate a board here.

The patch adds nine lines to the exact GPU trainer at
`4b60a882b1569f5a1a273a20c492c3510f703c6a`. It adds an optional, default-off
`--recover-erased-complete-coverage`. CompleteCoverage already draws its inputs;
the enabled branch preserves rows `[0:8,32:44]` immediately after that stage,
then copies them to `[16:24,56:68]` after US/factorized replacement and before
the original permutation. It keeps the original 128 inputs, labels, weights,
all sampler calls, random states, paired40, loss, optimizer and learning-rate
schedule. Only the selected 20 feature rows change. The training report records
the option. Existing argument gates require the full/factorized recipe.

Source identities:

- Original trainer: `dbd4e976c2c5f06e9b2a318aba7995cc2740a8e306bbab9fe0c8a26f1822587d`.
- Patched trainer: `2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904`.
- Patch: `aefc13520ff4f07fe0b343ec8e27f1358862f9bf3475db524b7ff5a1ff06fac7`.

Mac applied the patch to an exact separate source snapshot and compiled the
result. Pruning only the new disabled branches, option declaration and report
field produces the same AST as the original trainer. The capture/comparison
scripts compile. **They have not yet run against the actual full trainer.**
Mac lacks at least the original British FullSourceBatch pack and original
features-gpu-ami-extra main/ambient inputs. No substitute inputs, shortened
packs or regenerated arrays are accepted as proof of the exact GPU integration.
The static preparation receipt is
`results/mac-complete-recovery-source-preparation-v1.json`.

## Run the actual two-update check on GPU's existing inputs

Run from the GPU project's normal directory, so existing relative input paths
continue to resolve. Copy source into a fresh, separate directory. Obtain this
Mac handoff's scripts/patch from its committed research branch in another
directory; `NYX_RECOVERY_TOOLS` below points to that directory's
`research/wakeword_p4/experiments`. No executing training source is edited.

```sh
NYX_RECOVERY_STAGE=$(mktemp -d /tmp/nyx-complete-recovery.XXXXXX)
mkdir "$NYX_RECOVERY_STAGE/base"
git archive 4b60a882b1569f5a1a273a20c492c3510f703c6a research/wakeword_p4/training | tar -x -C "$NYX_RECOVERY_STAGE/base"
cp -R "$NYX_RECOVERY_STAGE/base" "$NYX_RECOVERY_STAGE/patched"
git -C "$NYX_RECOVERY_STAGE/patched" apply "$NYX_RECOVERY_TOOLS/recover_erased_complete_coverage_v1.patch"
```

Create two JSON config files in that stage. The original/default-off configs
are the same file. The enabled config is identical except for the final flag.
The existing full training paths below must remain original and hash-valid;
adjust only their directory prefix if inputs live elsewhere.

```json
{
  "trainer_args": [
    "--device", "cpu", "--threads", "2",
    "--paired-objective", "phrase-ranking",
    "--paired-rendering-coefficient", "1",
    "--paired-draw-noise-condition", "fixed",
    "--upstream", ".local/wakeword/microWakeWord",
    "--features", ".local/wakeword/features-gpu-ami-extra-v1",
    "--architecture", "teacher60", "--context-frames", "150",
    "--batchnorm-policy", "frozen",
    "--initial-weights", ".local/wakeword/mww-pretrained-reconstruction-v1/reconstructed.weights.h5",
    "--ambient-sampling", "stratified", "--prefix-negatives", "0",
    "--source-exponent", "0",
    "--full-source-pack", ".local/wakeword/british-training-features-v1",
    "--factorized-coverage-pack", ".local/wakeword/factorized-training-features-v1",
    "--us-temporal-coverage",
    "--us-coverage-pack", ".local/wakeword/us-training-features-v1",
    "--negative-container-mixture", "--positive-full-timeline", "--negative-full-timeline",
    "--complete-coverage-pack", ".local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study",
    "--steps", "120000", "--eval-interval", "5000", "--event-validation"
  ]
}
```

Call it `control.json`; create `enabled.json` with
`"--recover-erased-complete-coverage"` appended to `trainer_args`. Then run the
following three separate processes and compare them. The tracer stops after
exactly two actual optimizer updates, before validation, checkpoint selection
or export. Keeping `--steps 120000` preserves the original first-two-update
learning rates. This does not launch a 120k training run. Use the original GPU
Python environment that imports TensorFlow, NumPy, soundfile and the pinned
upstream. CPU mode avoids occupying the training GPU; schedule CPU/disk work
when it will not interfere with the current formal run.

```sh
python3 "$NYX_RECOVERY_TOOLS/capture_complete_recovery_integration_v1.py" \
  --trainer "$NYX_RECOVERY_STAGE/base/research/wakeword_p4/training/train_stream.py" \
  --config "$NYX_RECOVERY_STAGE/control.json" --output "$NYX_RECOVERY_STAGE/original"
python3 "$NYX_RECOVERY_TOOLS/capture_complete_recovery_integration_v1.py" \
  --trainer "$NYX_RECOVERY_STAGE/patched/research/wakeword_p4/training/train_stream.py" \
  --config "$NYX_RECOVERY_STAGE/control.json" --output "$NYX_RECOVERY_STAGE/disabled"
python3 "$NYX_RECOVERY_TOOLS/capture_complete_recovery_integration_v1.py" \
  --trainer "$NYX_RECOVERY_STAGE/patched/research/wakeword_p4/training/train_stream.py" \
  --config "$NYX_RECOVERY_STAGE/enabled.json" --output "$NYX_RECOVERY_STAGE/enabled"
python3 "$NYX_RECOVERY_TOOLS/compare_complete_recovery_integration_v1.py" \
  --original "$NYX_RECOVERY_STAGE/original" --disabled "$NYX_RECOVERY_STAGE/disabled" \
  --enabled "$NYX_RECOVERY_STAGE/enabled" --output "$NYX_RECOVERY_STAGE/comparison.json"
```

The intended check authenticates exact original/default-off inputs, RNG state,
losses and final variable bytes; identical initialization across all three;
unchanged 108 main inputs and all paired40 inputs with recovery; exact saved
20-row placement before the same permutation; labels/weights and every captured
RNG unchanged; first-update paired BCE/ranking unchanged and main BCE recorded;
two finite optimizer updates with frozen BN and changed enabled weights.
It observes actual main128/paired40 network tensors, not a tag substrate or
partial 20-window loss. The existing training modules compute the gradients.

Any exception is a failed/partial probe, not a pass; preserve that fresh output
and diagnose before another attempt. Send configs, preparation/source hashes,
three complete probe directories and the comparison result through Issue #13.
No checkpoint, TFLite, final-set, playback, microphone or device output belongs
in this probe. A passing integration still does not demonstrate quality or
authorize changing a running experiment; a future matched full training and
the existing development/independent acceptance remain necessary.
