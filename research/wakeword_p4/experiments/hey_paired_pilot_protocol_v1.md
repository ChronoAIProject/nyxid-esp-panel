# Hey Willow / Hey Nyx matched synthetic pilot v1

Status: protocol preparation; no new material generated or scored. This is a
separate experiment, not an amendment of the running CompleteCoverage formal.
Source request: Issue13 comments6859235962,6859742947,6860346609.

## Question and comparisons

First compare official `wn9_heywillow_tts` against our method trained on exactly
Hey Willow. Then compare our Hey Willow and Hey Nyx arms under matched recipes.
The first contrast still confounds architecture, frontend and proprietary
training material. Willow has more syllables than Nyx; the second contrast
estimates a phrase effect under this synthetic recipe, not intrinsic difficulty.
Hi ESP remains a same-input chain reference, not the primary matched-word arm.
Muse is optional and requires a separate protocol/runtime admission.

Mac owns authenticated official artifacts, factory dispatch, native P4 replay
and later complete-device validation. GPU owns explicit target configuration,
material provenance, paired synthesis, training and waveform INT8 replay.
No physical board, human voice or listening gate blocks GPU preparation.

## Execution order and locks

1. Finish the original120k sequence and original audits/PTQ/postprocessing without
   edits to284pins or checkpoint selection. Preserve all failures.
2. Implement a separate explicit target-config adapter for the existing training
   entry, with fail-closed checks that target labels and feature manifests agree.
   The current entry is not yet certified for arbitrary target words. Do not
   point Willow at old Nyx positive packs. Verify CPU batch/label integration
   before allocating a new GPU run.
3. Before synthesis, push a material lock containing exact TTS engine/version,
   model/config bytes, phonemizer/version, pronunciation inputs, voice-role
   assignment, text lists, commands, seeds and renderer/RIR hashes. Availability
   and encoded/PCM overlap checks must pass. No substitution after seeing scores.
4. Before training, push both full source/tensor schedules, initial weights,
   optimizer/frontend/export settings and code hashes. Before scoring, push
   candidate model hashes and the scorer/calibrator/runtime arithmetic contract.
5. Run paired pilot, then locked development calibration and complete curves.
   Do not enlarge or replace this pilot's data after seeing failures. Any larger
   study is a separately registered experiment.

## Pilot material and training recipe

Two targets: literal `Hey Willow` and project pronunciation input `Hey Nicks`
for the Hey Nyx label. Preserve those exact strings plus any engine-native
phoneme output; do not claim phonetic truth from the text. Use one fixed Piper
version and two fresh training voices, a third fresh development voice and a
fourth reserved acceptance voice. Exact identities must be locked in step3;
Joe/Mike and all historical development/final voices are excluded. If distinct
available voices cannot satisfy this, record failure rather than reuse them.
Shared TTS parent/pretraining exposure remains an explicit limitation.

Per training voice/target: standalone plus three fixed command carriers
(`turn on the light`, `what time is it`, `set a timer`), three speed factors
(.9,1,1.1), ten fixed render seeds:120 base positives/voice/target,
240 base positives/target across both voices. Each base render receives all
three noise conditions, giving720 training positives/target after augmentation.
Match voice, carrier, speed, seed and environment across target arms. Use common
background/noise and full RIR at quiet/25dB/15dB SNR, never truncate the phrase.
All source PCM retained; render hashes, nonempty/finite/format/sample counts,
clipping, full-tail retention, native phoneme/duration availability and rendered
boundary mapping must be audited before feature extraction. Unavailable native
phonemes/durations are recorded as unavailable, not inferred ground truth.

Small pilot: teacher60/context150, frozenBN, same pretrained initialization,
FP32/noTF32, seed726, batch128,20000updates/arm, eval5000. Use identical optimizer
and frontend settings serialized from the authenticated current recipe in the
training lock. Checkpoint is fixed final20000, not best development score.
No old target-specific paired/CompleteCoverage pack is silently inherited.
Both arms use paired common negatives and reciprocal whole other-target
negatives; all schedule differences must be solely target-dependent material.
Training remains disabled until exact negative inventory and schedules are locked.

## Labels and evaluation

Reciprocal targets are negatives: Hey Willow is negative for Nyx and Hey Nicks
is negative for Willow. Shared common confusions: `Hey`, `Hello`, `Hey William`,
`Hey pillow`, `Hey Nick`, `Hey next`, label `Hi ESP` rendered literally as
`Hi E S P`, and `Hey Muse`. Store labels separately from exact renderer inputs,
including the Hey Nyx -> `Hey Nicks` alias. Preserve per-phrase
and per-source results; no relabeling based on model responses.

Development: new third voice, same4carriers/3speeds/3noise conditions, separate
render seeds and RIR: exactly one render seed per carrier/speed/noise cell,
explicitly enumerated in the material schedule before synthesis. Runtime replay
uses the saved PCM at its original sample rate without another speed change.
Each matched target has36positives, each common confusion
has36negatives; reciprocal-target positives also form cross-target negatives.
Include complete fresh continuous negative recordings totaling at least8hours,
source-disjoint from training and every prior development/final inventory.
Reserve a different fixed>=8hour negative inventory and fourth voice for
acceptance. Lock exact inventories and encoded/PCM hashes before scoring.
Neither dataset exists/admission is claimed by this document.

Compare recalls at a shared nominal one-sided95Poisson FAR upper bound<=.5/h,
not a shared numeric threshold. Save every eligible/ineligible curve point.
Custom UINT8 detectors: version `uint8-sum-binary64-v1`, five raw bytes summed
as integers then binary64 /1280; threshold binary64,67step refractory and
half-threshold rearm, no epsilon. Implement/verify this before scoring; this
protocol does not activate it in any existing experiment. Custom grid remains
original138points. Official Willow uses actual API getter-confirmed working
points under the official contract below, never assume equivalent score scales.

Adopt Mac proposal6860757149:13 official configurations, default-no-set plus
requested {.4,.5,.6,.644,.7,.8,.9,.95,.98,.99,.999,.9999}. Requests enter the API
as binary32, word_index=1; record exact binary32 request and actual getter with
at least9 significant digits plus exact binary32 bits. Require setter return1;
request input uses public .4.. .9999 range, but actual getter is separately
validated finite with0<getter<1 and bits/text consistency. Do not force getter
into setter input range or equality with the request (Mac clarificationd0c589b). Retain failures and duplicate/quantized actual working points;
do not turn API failures into recognition misses or count duplicates as distinct
actual points. This is a locked contract, not evidence setter support has run.
Official events preserve detect() values:1=word,0=no detect,-1=channel verified;
no custom smoothing/refractory/rearm. Native per-clip initialization currently
uses63 zero-prime calls of512samples, complete input and zero-padded final
partial call. Future scored runs must retain prime results as well as input
events. Input offsets are min(call*512,N), with prime indexed separately.
Continuous backgrounds run whole without slice resets. Authenticate factory
model name/engine/mode/chunk, samples/CRC/calls and events. Native P4 and classic
QEMU are separately identified runtimes; neither timestamps nor getters are
interchangeable without same-PCM certification. Speech scoring remains gated
on actual runtime/setter certification.

For custom development eligibility require macro>=.8, standalone>=.75,
command>=.75,0common/cross-target events and FAR upper<=.5/h. Select maximum
macro, then lower FAR upper, then higher threshold; null if none. Official
reference curves are diagnostic even if no point meets all gates. Report full
phrase hit and early events separately; latency is from saved generated full
phrase boundary, including command-continuation cases, not human final-phoneme
truth. Command-container signed offsets are separate diagnostics, not wake-word
latency: without an authenticated wake boundary that latency remains unverified.
Preserve misses, latency distributions, per-noise/speed/voice results,
raw UINT8/official traces, reset/CRC/call-count evidence and all model identities.

## Acceptance and scope

Only a development pass may unlock this study's separately reserved acceptance;
freeze the selected candidate/threshold first, score it once alongside frozen
v30, official Willow and Hi ESP on the same locked protocol. v30/HiESP are
context references with their own target-positive labels, not mislabeled Nyx
or Willow detectors. Exact reference-positive material must join the material
lock. Acceptance uses the same positive/confusion/FAR gates, no retry, rescue,
extension or threshold reselection. Failed pilot retains v30 and all evidence.
Original Mike/test6/7 and three consumed final sets remain untouched; this
study cannot recycle or rename the original acceptance package.

No pilot result alone promotes a model. Real-waveform streaming INT8 checks,
independent synthetic acceptance and documented scope are required for the
GPU goal; Mac owns subsequent P4/resource/full-device claims. No synthetic
result establishes human generalization or human speech latency parity.

The .8/.75 gates are pilot engineering admission only, not official parity.
Report official-versus-custom Willow paired recall differences and uncertainty
at the common FAR target, including per-source and command results. Before
opening acceptance, lock reference materials, operating-point selection rules
and the paired interval method; do not infer parity merely from gate passage.
Mac consumed Samantha/Daniel12-clip native pilot10402b9 is diagnostic history,
excluded from fresh training/development/acceptance.

## Stage-specific synthesis lock (before any new synthesis)

4090-hey-paired-generation-plan-v1.json enumerates native and three-condition
render requests, exact voice/config/card, phones/tokens, versions, phonemizer
binary/data hashes, explicit seeded ONNX derivation and renderer/quality code.
It authorizes only training native/render generation after the exact plan is
pushed. Development/acceptance requests are reservations, not authorization
to generate or score them. Continuous background inventories, new source
encoded/PCM overlap, tensor schedules and full training lock remain required
before training; this stage does not certify overall material admission.
Training common confusions use the same four carriers/three speeds/ten draws
as targets. Native training2400requests ->7200rendered clips; each target has
720rendered positives across two voices. Dev/acceptance each reserve120native
requests ->360rendered clips (36per phrase); neither set is generated now.
Derived ONNX changes only the two RandomNormalLike seed attributes, leaving
learned tensors unchanged; retain original model SHA and derived SHA per draw.
This explicit new derivation is separate from all frozen historical models.
Any native quality failure disqualifies the full planned training pack; no
substitution, selective filtering or retry. No wake detector is invoked.
