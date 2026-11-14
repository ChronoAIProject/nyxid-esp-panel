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

## Completed original metadata census on GPU and Mac

GPU `7da71cda` ran the unchanged `28eb5a87` reducer on the original metadata,
exit 0. Publication `460bcecc` supplied the four original metadata/provenance
files. Mac anonymously downloaded the 22,211,837-byte archive, checked SHA-256
`fe15ab839bb6df3f17b264f6b84bd48f5d6199f7665bbac4aec75ddef24d0e9b`,
the exact member set and all four original payload hashes. The intake receipt
is `../results/mac-original-broad-training-metadata-intake-v1.json`.

Mac then ran the same pinned reducer on these authenticated original bytes,
exit 0. The 56,213-byte output is byte-identical to GPU's original result:
SHA-256 `c2cd420a6dc501f9b60aa4172e8976b05438dfb562914e0e6be5e9454bcb66fe`.
Result: `../results/mac-hey-broad-original-metadata-sampling-mass-v1.json`.
Both targets have own 25,128, reciprocal 25,128 and common 56,700 rows. All
common-negative carriers are bare. Reciprocal command negatives are present.
Expected main positives are 16 bare and 16/3 for each of light/time/timer;
prefix negatives have four slots per carrier. These remain sampling
expectations, not an observed draw or crop-content census.

The join in `join_original_training_carriers_and_errors_mac_v1.py` authenticates
the already-consumed development manifest and matches the original persistent
false-event records. Of 13 Nyx float errors at every registered point, two are
bare, four carry "turn on the light", one "what time is it", and six "set a
timer". Each development carrier has 72 common clips. All three erroneous
literal phrases already have 6,300 bare training rows each; their command
combinations have none. See
`../results/mac-original-training-carrier-and-persistent-error-join-v1.json`.
This makes the coverage gap concrete, but does not prove causality: two bare
errors remain, all development clips use one Semaine voice, and the variants
are correlated. No data, label, boundary, threshold or model was changed.

```sh
python3 research/wakeword_p4/experiments/join_original_training_carriers_and_errors_mac_v1.py \
  --repository-root REPO --development-packet ORIGINAL_REUSED_DIAGNOSTIC_PACKET \
  --census MAC_ORIGINAL_METADATA_CENSUS.json --report NEW_JOIN.json
```

GPU `460bcecc` also independently recomputed the saved 135-stream events with
its own streaming FSM. The Mac source/receipt review matches its auditor hash,
original full540 result and all 540 original NPZ identities to the already
authenticated Mac packet. See
`../results/mac-gpu-training-event-audit-source-review-v1.json`. This is source
and identity review of the remote execution, not a new local execution of the
GPU auditor. The independent counts agree with Mac's completed event run.

## Original metadata census interface

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

Retain the unmodified result and exact command/session/exit. An observed draw
census still needs the original source/window trace, excluding overwritten
slots and matching the retained tensor schedule. This tool does not supply
that evidence, measure pronunciation or window content, change the sampler,
justify a candidate promotion, or authorize opening fresh acceptance data.
