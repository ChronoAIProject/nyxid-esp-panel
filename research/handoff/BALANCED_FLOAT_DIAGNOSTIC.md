# Frozen v46 all-window representation diagnostic

Prepared after v47's complete rejection at GPU `6bb87cf`, on2026-11-12.
Coordination and the decision to do this diagnostic are in
[Issue13](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13#issuecomment-6815817541).
This is a measurement protocol, **not** another training recipe or acceptance
test. No loss, coefficient, threshold or model is selected from it.

## Fixed inputs and question

Use the rejected original selected10000 v46 weights, original TFLite and
training report. Their exact identities are fixed in
[`measure_balanced_float_representation.py`](measure_balanced_float_representation.py).
The original selected5000 v47 and its later report have different training
trajectories; they are not substituted into this v46 measurement.

Authenticate all40320 raw uint16 NPY files against the original factorized
report SHA `7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44`
before any model inference. This retains all8064sources,672conditioning IDs,
three speeds, four phrases and five conditions. The raw NPZ is not required.
Generated source containers do not certify phonetic endpoints or independent
human speakers. These are previously examined training sources.

The question is whether noise/RIR displacement and near-word separation are
visible in the **actual full300dimensional Dense input** and its fixed sigmoid
classifier. Preserve the ordered five-by-sixty input; do not temporal-average,
normalize it before measuring its norm, pick a score maximum, or shift each arm
to its own generation end. RIR extends its container60ms.

For each ID/speed group, use **every common absolute** end across its20cells:
`e % 3 == 0`, `max(feature_crop_end) <= e <= min(350, min(raw_length)-1)`.
Extract the known phase1 full window `[e-149:e+1]`. Every2016group has a legal
common interval. This follows the existing stride-three model, without changing
source PCM or a deployed audio pipeline.

## Measurements and weighting

For each phrase compare quiet to each other arm. For each arm compare Hey Nyx
to each of Hey Nick/Next/Mix, always at the same absolute end. Retain Euclidean
distance, cosine distance, both original vector norms, actual classifier-logit
change, and displacement parallel/orthogonal to its fixed weight vector.
Also retain original probability and logit summaries for each cell. Positive
near-word `logit_delta` means the near word's logit exceeds Nyx's; it is not an
event or a measured FAR.

Zero-vector cosine and zero-head projections are undefined and explicitly
counted, without adding an epsilon to invent a direction. Average defined
windows within each group, then the three speed means within an ID, then IDs
equally. Record undefined windows, partial groups and IDs lacking a defined
speed; no confidence intervals or independence claims are attached. All group
arrays and their hashes remain available for reanalysis, including every time
position and all300components.

If the original SavedModel is supplied, first compare its complete nonstate
variable inventory against the loaded H5 graph and check all60traces from the
first numeric training ID, including every speed/phrase/arm. Reset all six
buffers per trace and repeat the first trace. Compare original-export scores
at every chosen common end against the H5 full-window probabilities, with fixed
`atol=rtol=2e-5`. This finite preflight is not universal export equivalence.
Current received SavedModel file hashes are recorded; its historical training
record did not previously bind those per-file hashes. Omitting SavedModel is
explicitly reported and cannot establish export parity.

Only two CPU inference threads are used. Output consists of20cell arrays per
group, classifier weights, full group summaries, an aggregate report and all
source identities. Neither feature extraction nor audio hardware is invoked.

## Reproduce

After independently verifying and extracting the original handoff archives:

```sh
/Users/auric/tmp/output/wakeword-macos-research/.venv/bin/python -B \
  research/handoff/measure_balanced_float_representation.py \
  --records /path/to/factorized-training-features-v1/report.json \
  --feature-root /path/to/parent-of-original-feature-directories \
  --model-dir /path/to/original-v46-model \
  --saved-model /path/to/original-v46-model/saved_model \
  --upstream /Users/auric/tmp/output/wakeword-macos-research/microWakeWord \
  --output /path/to/fresh-diagnostic-output
```

The feature root contains the original sibling directories
`piper-balanced-training-features-v1/artifacts` and
`balanced-rendering-factorial-v1/artifacts`. Paths in the authenticated report
are preserved. Model directory contains original `training.json`,
`best.weights.h5` and `hey_nyx.tflite`.

The separate
[`probe_balanced_float_pipeline_mac.py`](probe_balanced_float_pipeline_mac.py)
uses only generated random inputs/weights and an original-format local export.
[Its completed result](../wakeword_p4/results/balanced-float-pipeline-smoke-mac-v1.json)
checks60traces, original-format variable inventories/resets, all257measurement
keys and aggregation. Five numeric/cohort tests exercise known distances,
undefined vectors, common-time boundaries, missing cells and equal-ID weighting;
all20handoff tests pass. These checks prepare the implementation and do not
replace inference on the original v46 artifacts.

Training-fit distances cannot establish generalization, causal explanations of
old development errors, improvement from an untrained representation loss,
Hi ESP parity, P4 latency/resources or long-call/reconnect stability. Existing
rejections, original event gates, unused-development/final boundaries and the
excluded dining-room production panel remain in force.

## Completed original-artifact measurement

The [original v46 Mac result](../wakeword_p4/results/v46-balanced-float-representation-mac-v1.json)
completed all40320raw authentications,2016groups,49031common endpoints and
980620classifier inputs. The original SavedModel variable inventory matches
the original H5. All60preflight streams match at1440common endpoints; maximum
probability difference is1.341104507446289e-6. Weights remained unchanged.

The separate [saved-array audit](../wakeword_p4/results/v46-balanced-float-array-audit-mac-v1.json)
uses [an independent arithmetic implementation](audit_balanced_float_arrays_mac.py),
without importing the measurement routines or running another model. It verifies
every2016array-file hash, full source/time geometry, all518112group-metric
sets and257equal-speed/equal-ID aggregate sets. Undefined counts are0. Maximum
summary difference is2.842170943040401e-14; reconstructed classifier probability
error is2.532690572820684e-7.

Under the frozen weighting, mean Hey Nyx logit is11.153621quiet and6.068187noise15
(paired delta-5.085434). Mean target-minus-Mix logit margin is28.229844quiet
and17.485718noise15; corresponding cosine distance is.960541→.872614. Noise25
does not show this same decrease (target mean12.052458); RIR-only mean10.950982
is near quiet. These are descriptive original-training results. They do not
identify a unique cause or establish a representation-loss remedy. They also
use different temporal scopes/weighting from the GPU two-endpoint diagnostic;
the numerical summaries are not claimed to be identical replications.

## Original v46 float export on reused Mac350

The [frozen follow-up](plans/v46-mac-float-gap-v1.json) and
[`measure_v46_mac_float_gap.py`](measure_v46_mac_float_gap.py) compare the exact
received SavedModel with the complete previously saved Mac350 INT8 regression.
Both original float features and the original INT8 input values dequantized to
float pass through that same SavedModel. PCM speed, all frontend hashes, prime,
tail padding, stride and the original `.99609375` threshold remain fixed.
This performs no new export, fitting or calibration.

The [completed result](../wakeword_p4/results/v46-mac-float-gap-v1.json) retains
189880float outputs in two modes against94940original INT8 outputs. Every mode
replays exactly twice after resetting all six state buffers; all31nonstate
variables remain unchanged. A sequential graph wrapper matches all241direct
signature outputs on the first complete source exactly in both modes.

All three paths classify the same12/50targets and5/90Nyx-near negatives at the
fixed historical threshold. The18INT8-zero target clips already have maximum
single-source float probabilities between3.15e-7and.0002764; all are below half
of one UINT8 output step (`1/512`). Their raw-float five-score maxima are at most
`.0001139`. The recognition gap is thus present before final output rounding
on these reused synthetic sources. This does not identify a training or
pronunciation cause, and does not make quantization differences negligible:
maximum pointwise float/INT8 residuals are.269840(original input) and.271839
(dequantized input); the input-rounding sensitivity maximum is.0104662.
Matching classifications at one fixed threshold are not general equivalence.

The [independent saved-array audit](../wakeword_p4/results/v46-mac-float-gap-v1-audit.json)
recomputes all189880outputs' hashes, `math.fsum` windows, events, per-voice/variant
counts and zero-cohort ranges without importing the producer's smoothing or
event policy. It passes and retains those residuals. No acoustic/P4, fresh
human, reserved-data, new-model acceptance or GPU-run change follows.

```sh
python research/handoff/measure_v46_mac_float_gap.py \
  --sources "$PUBLIC_MAC_PAIRED" --model-dir "$ORIGINAL_V46_MODEL" \
  --int8-study "$V46_MAC_REPLAY" \
  --plan research/handoff/plans/v46-mac-float-gap-v1.json --output "$FRESH_FLOAT_GAP"
python research/handoff/audit_v46_mac_float_gap.py \
  --float-study "$FRESH_FLOAT_GAP" --int8-study "$V46_MAC_REPLAY" \
  --output "$FRESH_FLOAT_AUDIT"
```
