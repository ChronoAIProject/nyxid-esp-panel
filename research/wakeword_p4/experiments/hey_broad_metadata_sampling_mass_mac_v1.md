# Original broad-training engine and carrier mass

The saved 135-stream event result separates good fit on selected training
witnesses from failures on the reused Semaine source. It does not establish a
unique cause. Before another training intervention, distinguish corpus counts,
expected sampler mass and observed surviving windows.

`audit_hey_broad_metadata_sampling_mass_mac_v1.py` has two deliberately separate
operations. Both pin GPU source `e1f341a9e2dc8dd073efd1f50e79d4912edbfed5` and the
original trainer/admission/schedule/paired-runner hashes. Neither reads audio,
model, development/acceptance data or feature arrays.

## Completed published-count derivation

The `--published-count-derivation` operation reads the published completed
generation counts, Qwen/Kokoro request plans, render plan and feature result. It
checks that three render variants reproduce 25,128 own-positive, 25,128
reciprocal and 56,700 common-negative rows per target. This is a derivation from
published reports and source policy, not an independent private-pack census.

Result: `../results/mac-hey-broad-published-count-engine-mass-derivation-v1.json`.

| Engine | Own source containers per target | Rendered own rows | Expected main positive slots / batch |
| --- | ---: | ---: | ---: |
| Piper | 8,064 | 24,192 | 32/3 |
| Kokoro | 240 | 720 | 32/3 |
| Qwen | 72 | 216 | 32/3 |

With the original exponent-zero engine weighting, an individual Qwen source
container has 112 times a Piper container's main-positive sampling probability;
a Kokoro container has 33.6 times. This is an explicit balancing policy, not
itself a defect or proof of the recognition problem. Source containers,
rendered variants and conditioning IDs are not independent people.

Original batches draw 48 ordinary negatives, then discard the first 16 through
prefix replacement. The remaining 32 ordinary negatives have expected common
mass 296/13 (about 22.77) and reciprocal mass 120/13 (about 9.23). Sixteen own
prefix negatives and 48 ambient windows complete the batch. Do not count all
96 source reads as 96 surviving source windows, or report every negative source
as a bare common-negative phrase.

```sh
python3 research/wakeword_p4/experiments/audit_hey_broad_metadata_sampling_mass_mac_v1.py \
  --self-test

python3 research/wakeword_p4/experiments/audit_hey_broad_metadata_sampling_mass_mac_v1.py \
  --published-count-derivation --repository-root REPO --report NEW_REPORT.json
```

The constructed checks cover unequal corpus sizes under equal-engine weighting,
the 16 discarded ordinary negatives, zero-labeled own-prefix windows and
invariance to scaling one engine's corpus. They pass on Mac. They are not an
execution of the original corpus census.

## GPU-side original metadata census, not yet executed on Mac

The default operation requires the original two target metadata files and
provenance receipts. It checks their hashes against the original published
feature result before grouping source role, engine, carrier and text. It emits
row/source/conditioning counts, exact rational expected slot counts and expected
loss-weight mass. It excludes discarded ordinary draws and distinguishes own
positive versus zero-labeled own prefix slots. Ambient source contents and
observed sampler windows are outside its scope.

Run from a worktree where the original GPU commit object is available:

```sh
python3 research/wakeword_p4/experiments/audit_hey_broad_metadata_sampling_mass_mac_v1.py \
  --repository-root REPO \
  --training-feature-root REPO/.local/wakeword/hey-broad-training-features-v1 \
  --report NEW_PRIVATE_CENSUS.json
```

Publish the unmodified result and exact command/session/exit. An observed draw
census still needs the original source/window trace, excluding overwritten
slots and matching the retained tensor schedule. This tool does not supply
that evidence, measure pronunciation or window content, change the sampler,
justify a candidate promotion, or authorize opening fresh acceptance data.
