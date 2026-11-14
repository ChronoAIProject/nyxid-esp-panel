# Train the wake word, rather than completion of the whole request

The causal candidate improves consumed recall, but its current training target
does not ask for immediate wake-word recognition in positive command examples.
This is a task-definition problem, independent of numerical matching.

At GPU commit745ee1e2, `build_causal_onset_material_v1.py` computes
`end = lead + source.native_samples + full RIR tail`. The native source includes
the whole text, including "what time is it" or "turn on the light".
`causal_sequence_supervision_v1.py` places the positive region30–270ms after
that whole-container end; the wake word and ensuing command are unsupervised.
The native generator explicitly records no phoneme durations. Consequently
this label cannot be described as a wake-word completion label.

The existing actual-primed replay makes this relevant to use: all5 detected
"Hey Nicks, turn on the light" variants fire1.23–1.33s after the generated
wake-prefix support, near the end of the complete request. Four others miss.
All6 false "Hey Muse, what time is it" triggers likewise occur near the end of
the request,1.29–1.59s after its prefix support. The time/timer positive
examples can fire much earlier. Thus32/36 overall detections alone hides a
connected-command latency weakness. The saved events and source observations
are in `../results/mac-causal-training-label-scope-v1.json`.

These generated boundaries are useful source coordinates, not measured phoneme
times. The evidence identifies a concrete target mismatch and actual late
events; it does not prove this is the sole cause of confusion or latency.

The minimal next practical experiment is to use the already available bare
"Hey Nicks." and "Hey Nicks!" positive sources for completion supervision,
keeping the existing negative/ambient material and causal runtime. This avoids
guessing a wake boundary inside an unaligned command and needs no new audio.
Keep the current failed model and its command examples for comparison; do not
crop audio or turn complete requests into keyword-boundary truth. Measure
early wake behavior, misses and false wakes directly after the next fixed run.
This is a different change from adding common-negative command carriers.

If positive commands are subsequently used for temporal supervision, supply
the wake-prefix boundary separately from the whole-container end. An actual
alignment or stored synthesis timing can supply it; word-count proportions or
the complete-request end cannot. Do not require exact CPU/GPU, gradient or
FP32/INT8 equality before testing the resulting usable model.

## Practical reference from the existing mature detector

The already completed actual P4 `wn10_himuse` test provides a more useful
connected-command reference than a requirement to match internal numbers.
On the same108 existing Semaine WAVs, at each model's already specified point:

| Behavior | Official Muse on P4 | Causal Nyx, actual-primed Mac replay |
| --- | ---: | ---: |
| Own target detected after source starts |36/36|32/36|
| Bare target |9/9|9/9|
| Target followed by a command |27/27|23/27|
| Light command |9/9|5/9|
| Reciprocal-word/Willow cross clips |4/72|6/72|

Muse's light detections occur about104–127ms after generated prefix support;
the five Nyx light detections occur about1233–1334ms after their support.
The source coordinates and different detector block clocks do not establish
human acoustic timing, but the practical delayed-command weakness is visible.
The comparison does not pool P4 and Mac execution into a hardware-equivalence
claim. Source results and limitations are retained in
`../results/mac-practical-muse-causal-reference-v1.json`.

The mature reference itself has four near-word false clips on this synthetic
set. Therefore “zero errors on every consumed near-word clip” must not be
presented as what the measured reference achieves. The candidate still has
more cross errors and substantially weaker light-command behavior here;
this does not make it qualified or erase its full-GPU common/background errors.
The intended comparison remains practical recognition, false wakes, timely
commands and eventual independent/P4 verification, not identical internals.

## Bare-positive experiment completed; connected speech still needs work

GPU commit89ce7259 reports the proposed minimal experiment's real completed
training, final INT8 export and actual-primed behavior. Do not repeat it as an
untried next step. At the existing0.91796875 reference the new model detects
28/36 targets, bare9/9 and commands19/27; light1/9, time9/9, timer9/9. Common
false clips fall from9 to1 and background events from1 to0 over the same
10.48hours. The one light hit is about223ms after generated-prefix support,
but8/9 light commands miss. Previous causal detected5/9 light commands late.
Faster surviving hits do not compensate for the extra misses; neither model
is ready to replace the mature detector.

The next useful direction is to retain full connected-command positives and
give them a separately estimated wake-prefix boundary. An existing phoneme
teacher or forced aligner can supply explicitly weak timing labels on original
training PCM, beginning with a small bare/light/time pilot on the two existing
probe speakers. Preserve unresolved examples as unresolved. Do not derive
boundaries from word-count proportions, whole-sentence ends or the wake
model's scores. Keep the complete command in the continuous training stream.
No new synthesis, audio crop or numerical agreement audit is required.
GPU's paired training-source behavior probe at04e1f9b1 should finish naturally
before selecting the precise intervention. Issue13 carries the current handoff.

## Aligned-prefix experiment completed; judge its practical tradeoff

GPU commitf9ad72b7 completes that intervention. All432 existing command sources
received automatic prefix references from the existing forced aligner; full
PCM remained in the training stream. The resulting60952-byte INT8 model
detects33/36 targets after source start, including9/9 bare and24/27 commands
(light6/9, time9/9, timer9/9) at the existing reference threshold. Light hits
are about103–193ms after the generated prefix reference, with a133ms median,
compared with the previous1273ms median. Three light commands still miss.

Near-word false clips increase to17/288;15 are Muse commands and2 are Hello
timer commands. Background events remain0 over the existing10.48hours.
These observations justify testing usability and improving discrimination;
they do not establish readiness or require a perfect-zero-error target.
Five bare detections precede the generated endpoint by3–33ms. Such offsets
are not themselves functional failures: the endpoint is a synthesis reference.

The user's priority is practical use, not numerical agreement. Do not repeat
CPU/GPU, FP32/INT8, gradient or saved-score agreement work as a gate. Use basic
artifact/ABI/inference-health checks and the existing simple streaming profile
to advance a promising candidate to development-board verification. Preserve
the independent final test, audio ownership and production-board boundaries.
