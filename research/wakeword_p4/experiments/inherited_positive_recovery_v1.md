# Recover inherited positive draws: preparation only

Current main32 and pair10 positive sources all request bare wake phrases. The
inherited original training metadata contains 34150 command variants from
3415 WAVs, but all original positive draws are overwritten. CompleteCoverage
recovery restores bare-phrase sources and does not fix that coverage gap.

The optional patch saves the already sampled/scaled original positive rows
24..31 before FullSource/Complete/US/factorized replacements, then restores
them to the same rows after those replacements. No new draw, RNG, batch,
label, weight, loss, command filtering, coefficient or firmware behavior.
The new flag defaults off and requires the existing factorized recipe.
Source snapshots are separate from the running GPU formal experiment.

This is recovery of the inherited **mixed source distribution**, not an
unconfounded command-only intervention. Original positives have 60000 Piper,
3000 Kokoro and300 Qwen rows. With current equal-engine draws, only Piper has
commands, 34150/60000; eight restored rows imply expected command count
`8 * (1/3) * (34150/60000) = 1.517777...` per batch. Actual counts must be
captured, not assumed or forced. It replaces eight British positive rows;
with Complete recovery on, none of its British positives remain, while its
British negative rows remain. No causal inference solely about commands.

Pins:

- Base trainer: `2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904`
- Patched: `b97b2debc92ca106ab199af60ac144aa7a78ce4a26e39621bc5b179fb637917e`
- Patch: `65e336cfec60d191748015d447e3c3f780ad4e82ac214160de3547a6297e1f09`

`prepare_inherited_positive_recovery_v1.py` authenticates/applies/compiles
the exact patch. Removing its optional AST nodes reproduces base AST. Fifty
symbolic batches per arm, with Complete recovery both off and on, verify
disabled transparency, restored original8, remaining120/main unchanged and
equal RNG, labels, weights, permutation. Source packs are mocked: this does
**not** establish real sampler/optimizer integration or acoustic quality.

## Required real-input integration

Preserve the current GPU sequence and its inputs/source pins. Run in a new
source/output directory on the machine holding original arrays. An independent
CPU two-update probe does not need to interrupt the GPU process. Keep the
same `steps=120000/eval-interval=5000`, initialization, fixed paired-noise
config and all original helper/input identities. The observer stops after
two actual updates, before validation or export.

The ready-to-run driver creates both independent source trees, applies the
patch, writes both flag combinations' configs, runs six CPU captures and the
two comparisons. Run from the GPU repository root so original `.local` input
paths resolve. It changes only common probe device/thread options; formal
configs and processes are untouched. Failure exits immediately with prior
outputs/logs preserved; use a fresh output if a diagnosed issue needs repair.

```sh
python research/wakeword_p4/experiments/run_inherited_positive_cpu_probe_v1.py \
  --gpu-ref 43254c0 \
  --control-config /path/to/original/fixed/control.json \
  --output /new/inherited-positive-cpu-probe
```

Create three contexts: exact base source, patched/default-off, patched/enabled.
Use the original same control JSON for the first two and append only
`--recover-inherited-positive-draws` for the third. For both Complete-off/on
combinations, run these in separate processes and outputs:

```sh
python capture_inherited_positive_integration_v1.py \
  --trainer /separate/source/research/wakeword_p4/training/train_stream.py \
  --config /separate/control-or-enabled.json --output /new/arm-output

python compare_inherited_positive_integration_v1.py \
  --original /new/original --disabled /new/disabled --enabled /new/enabled \
  --output /new/comparison.json
```

Capture all original32 source IDs/texts/actual crop ends before replacement,
original scaled features and true optimizer main128/pair40 inputs. Verify
default-off inputs/state/loss/final variables exact; identical initialization;
original8 reach actual optimizer; other120/main and pair40, labels/weights,
permutation/RNG and sampled source IDs unchanged; finite updates/frozen BN;
enabled final weights different. Report actual restored text/engine counts,
including zero-command two-step probes. Do not regenerate or filter to force
command presence. Keep failed outputs and logs.

The new capture/comparator are source-prepared, not locally real-input tested
because the Mac still lacks original arrays. GPU should return all captures,
configs, source SHA, comparison and logs. Do not start another120k run from
this document: current matched results and a separate frozen next-training
protocol/selection/development contract must come first. No consumed
development or existing finals enter training. v30/default Hi ESP unchanged.
