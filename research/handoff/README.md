# Hey Nyx research handoff for an NVIDIA RTX 4090

Snapshot: 2026-11-10. Branch: `research/hey-nyx-4090`.
This branch combines the complete Mac training research and the experimental
full-panel runtime. Track the independent
[GPU branch progress](GPU_PROGRESS.md) alongside each new experiment cycle.
Production still uses **Hi ESP**. Hey Nyx is **not accepted
as an equivalent replacement**. The dining-room production panel is excluded.

## Start on Linux

```sh
git clone --branch research/hey-nyx-4090 https://github.com/ChronoAIProject/nyxid-esp-panel.git
cd nyxid-esp-panel
uv venv --python 3.12 .local/4090-env
uv pip sync --python .local/4090-env/bin/python research/handoff/requirements-gpu.lock
.local/4090-env/bin/python research/handoff/gpu_smoke.py
python3 research/handoff/fetch_assets.py --output "$PWD/.local/research-snapshot" \
  --include-audio --include-history
export WAKE_DATA="$PWD/.local/research-snapshot/workspace/hey-nyx-training-v3/embedding-feasibility"
```

The CUDA lock resolves for Linux x86_64 / Python 3.12; this Mac has no NVIDIA GPU.
`gpu_smoke.py` must pass on the destination before GPU training can be called
verified. It checks GPU discovery, three random-data fitting steps, INT8 export
and BUILTIN_REF execution. It does not measure wake recognition. Install a
host driver compatible with the locked CUDA 12 packages. Native Windows should
use WSL2/Linux for TensorFlow GPU.

The original asset set is 1.19 GB compressed (about 1.11 GiB); the subsequent
low-level feature and model extensions add about 129 MB.
The ordered-context continuation adds another 7 MB of checkpoints.
All downloads have archive and per-file SHA256 verification. Default download
contains features/models; `--include-audio` also retrieves the synthetic source
WAVs, and `--include-history` retrieves historical experiment source. No key is
needed to continue fitting from these features. The historical filename
`jobs-private.json` contains only generated/public dataset provenance in this
export; it contains no device credentials or private room recordings.

## Reuse original speech with a different frontend

The existing downloaded snapshot with `--include-audio` contains six synthetic
TTS sources. `waveform_inventory.py` verifies the base original WAVs against
both the downloaded archive contents and their historical jobs/metadata,
checks mono16kHz PCM16 headers, and preserves source text, source-PCM keyword
endpoints and engine-qualified voice tokens. It rejects cross-split source
hashes/voice tokens. It never scores a detector or changes audio.

```sh
python3 research/handoff/waveform_inventory.py \
  --snapshot .local/research-snapshot --output .local/new-waveform-inventory
```

Actual verification on the extracted public snapshot covers 11,604 originals:
10,070 training and 1,534 validation, across Piper, Kokoro, Qwen, OpenAI,
macOS and Edge TTS. No source WAV hashes or voice tokens cross these splits.
The complete generated `waveforms.jsonl` has SHA256
`32a67247f94e71d2602e8ca338cc56ba4ae7e4ba634a39af67ef77896a83b95c`;
the sanitized report is
[`embedding-portable-waveform-inventory-v3.json`](../wakeword_p4/results/embedding-portable-waveform-inventory-v3.json).
`waveform-inventory-v1.tar.gz` publishes this portable manifest/report; raw WAVs
remain in `synthetic-training-audio.tar.gz`. Fetch a named incremental asset
into a fresh directory, then merge its verified workspace as usual.

This is an import inventory, not an evaluated new model. Existing development
voices and correlated variants do not provide fresh acceptance. Training
must preserve source partitions; Piper blends and Qwen description conditions
are not independently established human speakers. The separate 960 extended
Qwen sources and all testing/private microphone recordings are excluded.
Use `keyword_end` as an approximate 16 kHz source-sample endpoint when rendering
the destination frontend; derived frame indices and old feature arrays are
deliberately omitted. The GPU 10 ms MicroFrontend and local 80 ms embedding cannot
interchange precomputed arrays. This allows the GPU track to reuse broader
source coverage without repeating TTS generation.

## Continue fitting from an existing checkpoint

The completed phase experiment adds training-only 1..1279-sample phase offsets
without changing speed, duration, original sources, class/engine sampling or
the held-out splits. Its matched control and treatment are fully trained and
calibrated. Reproduce the treatment from its original warm start:

```sh
export WAKE_PLAN="$WAKE_DATA/portable-plans/head-temporal-phase-phase-plan-v1.json"
.local/4090-env/bin/python research/wakeword_p4/embedding/train_head.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_PLAN" \
  --initial-weights "$WAKE_DATA/head-temporal32-dropout-v1/step-20000/head.keras" \
  --output "$WAKE_DATA/new-4090-phase-reproduction"
.local/4090-env/bin/python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_PLAN" \
  --training "$WAKE_DATA/new-4090-phase-reproduction"
```

Use a fresh output directory for every experiment. These scripts warm-start
model weights; they do **not** restore the original optimizer or RNG position.
For a new recipe, copy a portable plan, record changes before fitting, bind the
chosen initial `.keras` SHA256 in `initial_weights_sha256`, then pass that exact
file with `--initial-weights`. The trainer verifies it. Preserve normal audio
speed and use the same fixed validation operating-point policy. GPU and CPU
training need not produce identical new weights; report the environment.

The source feature indexes from two older directories were already relative.
Newer extra-view indexes contained absolute Mac paths: the exporter makes only
their `path` values relative and generates plans binding those new index hashes.
`original-index.jsonl` and `portable-feature-provenance.json` retain their original
identities. **Array bytes, source labels/splits, model weights and recipes do not
change.** A complete equal-mean calibration replay from the extracted directory reproduces
the original calibration JSON exactly. Original calibration and checkpoint files remain unchanged and refer
to their original audit identities. Use the portable plans for new fitting;
do not rewrite old calibrations to pretend that their metadata is unchanged.
Original absolute `jobs[].source` paths are corpus/audit metadata for fitting,
not live dependencies. Raw-feature regeneration needs paths rebased explicitly
to the downloaded workspace or public datasets retrieved from their manifests.

## Current evidence and useful starting points

| Candidate / test | Result | Interpretation |
| --- | --- | --- |
| Frozen equal mean of DNN and temporal heads, threshold .4 | Hey Nyx 46/50 paired variants; Hi ESP 48/50 | Existing development audio, not fresh human acceptance |
| Same candidate ordinary negatives | 1/100; Hi ESP 0/100 | Remaining false wake gap |
| Same candidate English continuous | 1 event / 5.549 h; Hi ESP 0 | Development regression |
| Same candidate Mandarin continuous | 0 / 1.294 h; Hi ESP 0 | Limited exposure, not a zero-FAR guarantee |
| Latest original-app Hi ESP acoustic control | Wake, immediate request, reply and quiet pass | Physical Mac-speaker control passed |
| Latest fixed Hey Nyx full-app acoustic repeats | 3/5 wakes; all 3 complete request/reply/quiet and rearm | Two misses remain; no replacement acceptance |
| Five-minute whole-app resource test | Mean 51.3 ms / 80 ms step; no new RX overflows; byte balance -640..0 | Passed these capture/resource gates only |
| Phase continuation control vs treatment | Validation macro recall 94.05% vs 94.29% | Small validation difference; no new regression/physical result |
| Phase continuation equal mean | Validation macro recall 94.73%, threshold .3 | Original frozen mean 94.68%; +0.0417 percentage points, without new physical acceptance |
| Low-level continuation, both heads | Validation macro recall 94.16%, threshold .4 | Matched continuation control 94.68%; both 1 false event / 7.304 h; rejected |

Detailed reports: [`../wakeword_p4/results/`](../wakeword_p4/results/), especially
`embedding-full-app-resources-v3.json`, `embedding-head-equal-mean-v3.json` and
`embedding-phase-continuation-v3.json`. Older failures remain in Git history.

### Subsequent acoustic diagnosis

An opt-in whole-app input trace now verifies the exact PCM consumed by the
detector, frontend bytes, embedding, both heads and event policy. In one home
development-panel run, all **1,682 steps match the host byte for byte**, with
5,046 complete fragments and no reported drops. Forty seconds of steady-state
resource observation passes: mean inference51.8ms/80ms, no new hardware RX
overflows, byte balance -640..0. Private room audio stays off GitHub.

At the same recorded Mac output level100, the Hi ESP control completes its
initial directive, one reply and eight seconds of quiet. Hey Nyx misses its
paired synthetic cue: peak .2181 at the frozen .4 threshold. This is one trial,
not a new recall estimate. The original Hi ESP app/service is restored. See
`embedding-input-trace-hardware-v3.json` and `embedding-home-acoustic-level-v3.json`.
The acoustic harness now rejects muted/zero-volume playback and records output
level before/after; earlier muted playback cannot establish a detector miss.

The new `prepare_level_views.py` experiment pairs low-level training views with
the completed environmental views. Existing augmentation targets RMS .04.. .20
of PCM full scale; the new prespecified range is .002.. .02. This is a hypothesis
about acoustic robustness, not a proven cause. Original room, background, SNR
and RNG draws must match the completed reference index. The input PCM speed,
source/held-out partitions and existing views remain unchanged. Plans under
`embedding/plans/level-*.json` freeze extraction and the fitting comparison.
All 20,140 training-only low-level views are complete. Every feature hash and
paired source/background draw was checked; the 24-view pilot agrees exactly
with the corresponding full extraction. The temporal-only continuation yields
94.10% validation macro recall in combination with the original dense head.
Continuing both heads yields 94.16%, compared with the matched control's 94.68%.
The fixed original selection policy chooses threshold .4 for both combinations;
both produce one false event in 7.304 h of validation audio. These results do
not establish an improvement, so these candidates were not flashed. No new
classifier or production wake replacement is accepted. The prior 95.26% figure
in this document referred to the temporal head alone, not the original equal
mean; the original result JSONs were already correct.

The asset manifest now includes these new features, all four temporal
checkpoints and both dense continuation arms. An existing snapshot can fetch
only the extensions into a **fresh staging directory**:

```sh
python3 research/handoff/fetch_assets.py --output .local/level-update \
  --asset level-continuation-v1.tar.gz --asset level-dense-checkpoints-v1.tar.gz
```

After verification, merge its `workspace/` into the existing snapshot. Extension
paths do not replace original feature/model directories. For new fitting use
`portable-plans/head-temporal-level-plan-v1.json` or
`portable-plans/head-dense-level-plan-v1.json` and the same initial weights as
their corresponding frozen comparison plans. New raw-feature extraction still
requires the original source paths to be rebased as described above. Detailed
results are in `embedding-level-continuation-v3.json`.

### Matched low-level stress diagnostic

The low-level continuation has a local gain at one stress level, without
consistent improvement over the original model. The diagnostic uses the 1,534
existing variant-1 validation clips: 598 positive and 936 confusing negative
sources. Every scale-1 feature and all three models' score sequences reproduce
the original stored values exactly. Scaling the complete rendered signal keeps
its original speed, room response and SNR; all thresholds stay .4.

| Signal multiplier | Original frozen mean | Matched continuation control | Low-level continuation |
| --- | --- | --- | --- |
| 1 | 93.02% | 93.02% | 92.10% |
| .05 | 89.80% | 87.92% | 90.90% |
| .1 | 91.28% | 89.49% | 90.76% |

These are macro recall percentages across TTS engines, not independent human
success rates. Confusing-negative clip results remain in the report; this
short-clip test does not measure continuous false events/hour. The scale-1 row
covers only variant 1, so it differs from the full validation metrics above.
No checkpoint or threshold is selected from this diagnostic. Replay it from the
downloaded snapshot (including synthetic audio) without changing bound files:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/evaluate_level_stress.py \
  --data-root "$WAKE_DATA" \
  --plan research/wakeword_p4/embedding/plans/level-stress-evaluation-v1.json \
  --source-prefix /Users/auric/tmp/output \
  --source-root "$PWD/.local/research-snapshot/workspace" \
  --output .local/new-level-stress --workers 3
```

The original corpus prefix is audit metadata. Only path resolution changes;
every source WAV must retain its bound SHA256. See
`embedding-level-stress-v3.json` for the complete fixed-threshold comparison.

### Ordered temporal context continuation

A further matched experiment tests an ordered Dense32 conjunction after the
two existing temporal feature stages, against a fixed uniform-mean control.
Both start from identical pretrained feature/classifier arrays; the new layer
starts as an exact temporal mean. All 512 training fixtures agree initially
within 5.96e-8. Re-running the full 10,000-step treatment from the isolated
research entry point reproduces all four checkpoints' weights and INT8 model
bytes exactly. `train_head.py` stays byte-identical, preserving older plan hashes.

| Combination | Validation macro recall | Paired wakes / 50 | English events / 5.549 h | Mandarin events / 1.294 h |
| --- | --- | --- | --- | --- |
| Original frozen equal mean | 94.68% | 46 | 1 | 0 |
| Context mean control | 94.50% | 45 | 2 | 0 |
| Ordered Dense32 | 95.00% | 45 | 3 | 0 |
| Hi ESP reference | Separate model/policy | 48 | 0 | 0 |

The Dense32 combination's validation-selected threshold is .3; control .4.
Neither threshold is changed after the fixed development regression. Both
have one false wake in 100 ordinary sentence clips. The small validation gain
does not persist in development regression, so both new candidates are
rejected and were not flashed. These are reused development datasets, not
fresh independent acceptance. Complete results and source hashes are in
`embedding-context-continuation-v3.json`, with initialization and full-training
replay reports alongside it. The motivating near-word error counts are in
`embedding-validation-errors-v3.json`; they suggest a hypothesis, not a cause.

`context-continuation-v1.tar.gz` adds both four-checkpoint families, the exact
initialization files, validation arrays/calibrations and portable plans. Fetch
it with `--asset context-continuation-v1.tar.gz` into another fresh staging
directory and merge its verified workspace into the existing snapshot.
Reproduce the treatment with its unchanged existing feature arrays:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/train_context_head.py \
  --data-root "$WAKE_DATA" \
  --plan "$WAKE_DATA/portable-plans/head-temporal-context-dense-plan-v1.json" \
  --initial-weights "$WAKE_DATA/temporal-context-dense-initial-v1.keras" \
  --output "$WAKE_DATA/new-4090-context-reproduction"
```

`prepare_context_initialization.py --help` documents regeneration of the matched
initializers from the original checkpoint and training representatives. Keras
ZIP metadata can change the regenerated archive hash; weight arrays match.
Use the exported exact initializer when replaying an existing hash-bound plan.
`regress_equal_mean.py` reuses the audited historical regression procedure with
an explicit data root. It requires the original paired/continuous WAVs, Hi ESP
reports and cache identities; these complete physical-baseline source folders
are not all included in the fitting snapshot. No private room audio is needed.

### Generic channel-EQ continuation

The completed training-only EQ extension adds 20,140 paired environmental
views from the same six engines. Independent five-band gains preserve each
original source/background draw, audio duration, rate and speed. It does not
fit or load a private microphone recording. All 43,730 original indexed arrays
remain unchanged; the balanced 48-row pilot/full replay and pre-EQ reference
features are exact. See `embedding-eq-views-audit-v3.json`.

Both existing dense/temporal heads continue for 10,000 steps from their exact
original initializers, with the same budget, class/engine sampling recipe,
PTQ representatives and validation policy as their recorded controls. All 512
PTQ representatives match each control byte-for-byte. Only the added EQ views
change their training feature sets; extra views intentionally change sampled
within-engine examples. The recipe and equal-mean combination are frozen
before fitting in `embedding/plans/eq-continuation-v1.json`.

| Combination | Validation macro recall | Paired wakes / 50 | English events / 5.549 h | Mandarin events / 1.294 h |
| --- | --- | --- | --- | --- |
| Original frozen equal mean | 94.68% | 46 | 1 | 0 |
| Generic EQ equal mean | 94.57% | 45 | 2 | 0 |
| Hi ESP reference | Separate model/policy | 48 | 0 | 0 |

The EQ combination freezes threshold .5 on original validation before
development scoring. It has one ordinary false wake / 100 sentence clips.
It is rejected and was never flashed. A post-calibration diagnostic on the
same previously missed private capture raises peak score from .2181 to .4915,
but both models still miss at their own fixed thresholds. This single reused
cue does not justify lowering a threshold or fitting a measured-room filter.
The aggregate acoustic diagnostic and full rejected experiment are published;
the recording remains private. See `embedding-capture-transfer-diagnostic-v3.json`
and `embedding-eq-continuation-v3.json` under `research/wakeword_p4/results/`.

`eq-continuation-v1.tar.gz` includes the new features, both four-checkpoint
families, exact initialization files, calibrations and portable plans. Fetch
it incrementally with `--asset eq-continuation-v1.tar.gz`, verify it, and merge
its workspace into the existing downloaded snapshot without changing existing
files. To replay one head with the same data and unchanged trainer:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/train_head.py \
  --data-root "$WAKE_DATA" \
  --plan "$WAKE_DATA/portable-plans/head-temporal-eq-plan-v1.json" \
  --initial-weights "$WAKE_DATA/head-temporal32-dropout-v1/step-20000/head.keras" \
  --output "$WAKE_DATA/new-4090-temporal-eq-reproduction"
```

For dense replay, use `head-dense-eq-plan-v1.json` and initializer
`head-environment-expanded-v2/step-10000/head.keras`. No private microphone
recording or physical baseline source folders are needed to fit these heads.
Portable training plans only rebind normalized feature-index paths; arrays,
labels, recipe, initialization and model bytes remain exact. These repeated
development results are research evidence, not fresh user acceptance.
The published archive and every member were downloaded/verified. Both heads'
first five batches agree exactly over 60,767 portable training rows, and the
portable equal-mean calibration reproduces the original JSON byte-for-byte.

The original 4090 bundle includes all then-completed `head-*` model checkpoints (DNN,
temporal, mining, Qwen, environmental, pooling and phase variants), original
MicroWakeWord model/weight artifacts from training v1/v2/v3, 61,064 indexed
embedding histories across eight completed feature views, synthetic source
WAVs, frontend coefficients, per-checkpoint numerical fixtures and validation score
arrays for exact equal-mean recalibration, plus the three
hash-bound board runtime bundles. See `assets.json` for exact sizes and hashes.
The selected runtime bundle is `runtime-equal-mean-bundle-v3`; its calibration
is `head-equal-mean-validation-v1/calibration.json`.

Do not conflate correlated renderings with independent speakers. The six TTS
engines are Piper, Kokoro, Qwen, OpenAI, macOS say and Edge TTS. Reserved final
English 6.455 h and Mandarin 3.126 h remain unscored and are excluded from this
training export. Public raw negative corpora can be downloaded again using
the source manifests/scripts; their ready-to-fit embeddings are included.
Build caches, installed dependencies, redundant old derived features, NVS,
credentials, device snapshots, private microphone captures and provisioned
firmware backups stay local.

### Confidence-bound operating-point diagnostic

A separate validation-only operating point keeps both original heads, equal
weights, three-step score averaging and all event timings unchanged. The
original rule constrained the observed continuous false-event rate to .5/h;
the alternative constrains its one-sided 95% Poisson upper bound to .5/h.
The rule was chosen after reviewing the existing original validation grid and
frozen before any new candidate development/acoustic score. The ongoing .4
physical experiment is a separate measurement and is unchanged.

| Original validation, same 7.304 source hours | Original point-estimate rule | Confidence-bound rule |
| --- | --- | --- |
| Threshold | .4 | .6 |
| Engine-macro positive recall | 94.685% | 92.801% |
| Continuous false events | 2 | 0 |
| One-sided95% upper bound, events/hour | .862 | .410 |
| Positive hit latency median / p95 | 130 / 234 ms | 176 / 275 ms |

This is a recall/latency tradeoff, not a proven improvement or independent
test guarantee. Threshold-grid selection, repeated experiments and limited
speech/environment coverage constrain the interpretation of the validation
bound. The candidate is not accepted or flashed. Physical acceptance and
resource/voice-chain gates remain pending; completed development regression
is reported below.
No threshold was selected from microphone trials or development regression.
The complete original confusion-group metrics remain in
[`embedding-confidence-operating-point-v3.json`](../wakeword_p4/results/embedding-confidence-operating-point-v3.json).

Reproduce the post-selection using the unchanged original public snapshot:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/select_confidence_threshold.py \
  --data-root "$WAKE_DATA" \
  --plan research/wakeword_p4/embedding/plans/equal-mean-confidence-v1.json \
  --output .local/new-confidence-calibration
```

The frozen regression protocol is
`embedding/plans/head-equal-mean-confidence-regression-v1.json`. It retains
the original corpus/helper identities and differs only in the validation
operating point and its explicit candidate identity. Preserve all old results
and use fresh output directories for this diagnostic.

The completed regression reproduces every original source/cache identity and
has no priming events. Increasing the frozen threshold trades recall for fewer
near-word/continuous events; it does not fix the known ordinary-sentence event:

| Existing development measurement | Original .4 | Confidence-bound .6 | Hi ESP |
| --- | --- | --- | --- |
| Positive clips with any event | 46 / 50 | 40 / 50 | 48 / 50 |
| Nyx-specific near-word clips with events | 23 / 90 | 15 / 90 | 0 / 90 |
| Shared ordinary-sentence clips with events | 1 / 100 | 1 / 100 | 0 / 100 |
| English continuous events / 5.549 source hours | 1 | 0 | 0 |
| Mandarin continuous events / 1.294 source hours | 0 | 0 | 0 |

The new zero-event 95% upper bounds remain .540/hour for English and
2.316/hour for Mandarin, computed from their slightly shorter evaluated
durations. Reused development corpora do not provide independent acceptance.
The .6 candidate is rejected as a replacement and was not flashed. Full
reports and paired phrase counts are in
[`embedding-confidence-regression-v3.json`](../wakeword_p4/results/embedding-confidence-regression-v3.json).
The .4 physical bracket below remains an independent, unchanged measurement.

### Matched physical five-voice pilot

The frozen original equal-mean app (threshold .4) was tested through the Mac
speaker, between original Hi ESP app measurements. All 30 short trials used
unchanged source PCM, the same output100/unmuted built-in speaker, and the same
last-confirmed approximately 0.5 m quiet placement. No microphone audio was
recorded. All wake calls were stopped immediately and every trial rearmed.

| Arm, in execution order | Positive triggers | Ordinary-negative triggers |
| --- | --- | --- |
| Original Hi ESP before | 1 / 5 | 0 / 5 |
| Original frozen Hey Nyx | 4 / 5 | 0 / 5 |
| Original Hi ESP after restoration | 2 / 5 | 0 / 5 |

Samantha changes from a Hi ESP miss to a hit across the bracket. Daniel misses
all three arms; Tessa hits all three. Keep the failed controls and this
variation explicit. The five already-used synthetic voices are correlated
development conditions, not fresh human acceptance. Short negative clips do
not establish continuous false events/hour. These whole-app paths also have
different detector inputs: default Hi ESP retains AEC/AGC, while experimental
Hey Nyx reads raw mic from the same producer. This is not a classifier-only
matched-input comparison or evidence of population parity.

The sanitized per-trial events, source hashes, conservative HTTP latency
intervals and recovery evidence are in
[`embedding-acoustic-matched-pilot-v3.json`](../wakeword_p4/results/embedding-acoustic-matched-pilot-v3.json).
The original Hi ESP app, service selection and idle listener were restored
and verified. No threshold, waveform speed/level or model was selected from
these acoustic trials. Conversation completion, echo and long-call acceptance
remain separate gates.

The small synthetic-only asset includes all 15 unchanged source WAVs, selected
source provenance and the exact frozen plan. Its public archive and every
member have been downloaded and verified:

```sh
python3 research/handoff/fetch_assets.py --output .local/acoustic-pilot \
  --asset acoustic-matched-pilot-v1.tar.gz
```

Use [`acoustic_wake_probe.py`](../panel_wake_runtime/acoustic_wake_probe.py)
on macOS with the documented guarded app-swap procedure. Linux can inspect the
same assets; its playback system needs a separately frozen protocol. Device
addresses, credentials and provisioned firmware are deliberately absent.

The subsequent frozen plan `acoustic-matched-extended-v1.json` expands each
arm to all 25 positive variant-0 phrases and 50 shared ordinary sentences from
the same five development voices. A fixed seed shuffles matched trial keys
identically across engines. It keeps the model, thresholds, normal speed,
playback settings and Hi ESP / Hey Nyx / Hi ESP bracket unchanged. Its full
100-WAV source asset is `acoustic-matched-extended-v1.tar.gz`, fetched with the
same `--asset` command into another fresh directory. The completed 225-trial
bracket is a prespecified development measurement, not a new independent
holdout; the smaller pilot remains separate.

| Expanded arm, in execution order | Positive triggers | Onset-ambiguous positive triggers | Ordinary-negative triggers |
| --- | --- | --- | --- |
| Hi ESP before | 6 / 25 | 0 | 0 / 50 |
| Frozen Hey Nyx .4 | 12 / 25 | 1 of the 12 | 0 / 50 |
| Hi ESP after restoration | 5 / 25 | 0 | 0 / 50 |

The Hey Nyx `Tessa-positive-2` HTTP interval straddles source onset, spanning
1.744–3.702 s; keep it ambiguous rather than count it as a confirmed wake after
speech. There are 11 Hey Nyx triggers with intervals clearly after source
onset. All 225 trials rearm with unchanged playback settings; the original
app, service and idle listener are verified after the final arm. Hi ESP
detects only Samantha/Tessa in both arms, and three individual Tessa outcomes
change across the bracket. Its low and variable control performance limits
interpretation of the apparent Hey Nyx advantage. The same preprocessing,
synthetic-voice, short-negative and latency limitations as the pilot apply.
See all retained trials and recovery evidence in
[`embedding-acoustic-matched-extended-v3.json`](../wakeword_p4/results/embedding-acoustic-matched-extended-v3.json).

### Backbone precision isolation

One already-viewed source/capture pair is replayed with unchanged native mel,
heads, .4 threshold, source level/speed and 16 kHz sampling. The three paths
are actual INT8 backbone, original float backbone with the identical
dequantized INT8 input, and float backbone with native mel. Float outputs
are requantized into the existing INT8-head space. The first nine shared
INT8 history steps are excluded from embedding differences; head scoring
starts only after complete float context. Actual INT8 extraction matches the
existing extractor byte for byte, including the known capture peak.

| Peak equal-mean score | Actual INT8 | Float, dequantized input | Float, native mel |
| --- | --- | --- | --- |
| Existing synthetic source | .4492 | .6712 | .6921 |
| Existing microphone capture | .2181 | .1198 | .1204 |

The source triggers all three paths, while the capture triggers none. This
does not support quantization alone as the cause of this particular missed
cue. It does not establish behavior on other recordings or accuracy of a
retrained float system. No fitting, new recording, model switch or float
deployment occurred. Only hashes and aggregate diagnostics are published in
[`embedding-backbone-precision-diagnostic-v3.json`](../wakeword_p4/results/embedding-backbone-precision-diagnostic-v3.json);
the microphone WAV, arrays and score sequences remain private.

To reproduce on the original local pair, supply its existing private paths;
the plan checks both WAV hashes and all model/helper identities:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/diagnose_backbone_precision.py \
  --data-root "$WAKE_DATA" --source "$WAKE_SOURCE_WAV" \
  --capture "$WAKE_PRIVATE_CAPTURE_WAV" \
  --plan research/wakeword_p4/embedding/plans/backbone-precision-diagnostic-v1.json \
  --output .local/new-private-precision-diagnostic
```

The private capture is deliberately absent from the public handoff. A new
pair requires a separately frozen plan and cannot reproduce this exact result.

### Portable paired PCM baseline

`paired-mac-regression-v1.tar.gz` supplies the complete previously scored
350-clip paired synthetic development set and its measured standalone-P4
Hi ESP results. It contains50 Hey Nyx,50 fully spelled Hi ESP and250
ordinary/near-word negatives, using Daniel, Karen, Moira, Samantha and Tessa
with the two existing normal-speed variants0/3. Variant renderings are
correlated; they are not new speakers or independent acceptance trials.

The exporter binds the original352-row manifest and restored Hi ESP report
by SHA256. It checks every selected WAV/header/sample count, decoded PCM CRC,
512sample inference coverage, event offsets and source/result match. Only
350 paired rows are selected; the two prior acoustic controls are not opened
or exported. Numeric results and WAV bytes stay exact, paths become relative,
and only explicit metadata/result fields are retained. No microphone audio,
UART/account/device state, provisioned firmware or frontend arrays are read.

```sh
python3 research/handoff/fetch_assets.py --output .local/paired-comparison \
  --asset paired-mac-regression-v1.tar.gz
export WAKE_PAIRED="$PWD/.local/paired-comparison/workspace/hey-nyx-training-v3/paired-mac-regression-v1"
python3 research/wakeword_baseline/compare.py \
  --manifest "$WAKE_PAIRED/manifest.jsonl" \
  --baseline "$WAKE_PAIRED/baseline.json" \
  --output .local/paired-hiesp-summary.json
```

The baseline is `wn9_hiesp`, `DET_MODE_95`, threshold.63:48/50 positive clips
trigger at any time,32/50 trigger in the approximate[-100,+800]ms keyword-end
window, shared ordinary sentences0/100, Nyx-specific near words0/90. Keep
these metrics separate. Hi ESP-specific near words retain pronunciation
uncertainty and are a different phonetic distribution. Standalone numeric
inference timing is not whole-app CPU or microphone/AEC acceptance.

GPU models should run their own frozen frontend/model/event protocol on the
same unchanged WAVs. A50-clip Hey Nyx count should be compared with the
50 Hi ESP positives; ordinary/shared phrase groups can use identical PCM.
Do not transfer any local80ms embedding cache or keyword-frame coordinates
to a10ms frontend. This is the reused development set; its outcomes must not
select thresholds or replace the independent final challenge. The public
download is archive/member-verified and the normalized Hi ESP summary is
replayed; see
[`embedding-portable-paired-baseline-v3.json`](../wakeword_p4/results/embedding-portable-paired-baseline-v3.json).

To rebuild the export on the original machine:

```sh
python3 research/handoff/export_paired_baseline.py \
  --manifest "$ORIGINAL_PAIRED_MANIFEST" --baseline "$ORIGINAL_HIESP_REPORT" \
  --plan research/handoff/plans/paired-mac-regression-export-v1.json \
  --output .local/new-paired-export
```

### Matched measured-room training study

The new `embedding/plans/measured-room-views-v1.json` freezes a separate
80ms embedding experiment before new detector scoring. Both arms add one
variant3 room view for each of the10,070 original training TTS sources.
Control reuses the exact completed synthetic-room array. Treatment replaces
only that response by one of270 hash-pinned public measured MIT responses.
The original room RNG draws are still consumed, preserving background/SNR,
bandpass, scalar level and all subsequent random draws. Both arms retain all
existing source/environment views and use matched additional-view counts.

Measured responses are aligned at the first sample within20dB of their peak
and scaled to the original per-job response's L2 energy. The existing renderer
keeps sample count,16kHz sampling, speech rate,2.4s lead,1s tail and keyword
timebase. Its fixed window can truncate longer reverberation tails. This is
an acoustic training hypothesis, not a fitted microphone response, proof of
causation, or a deployment-room coverage claim. It differs from the GPU's
10ms MicroFrontend room experiment; their derived features are not shared.

All270 response WAV hashes and headers were verified against the GPU branch's
public pinned audit (`4c3d8370…`). Raw responses remain local. Attribution:
MIT Computational Audition Lab,
[environmental impulse response survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html),
using the pinned16kHz mirror identified in the plan. Fetch with the research
NumPy/soundfile environment:

```sh
curl -fL https://raw.githubusercontent.com/ChronoAIProject/nyxid-esp-panel/2f70cc0fac55979700080d8f1c1dce867132f96d/research/wakeword_p4/results/4090-mit-rir-audit.json \
  -o .local/mit-rir-audit.json
.local/4090-env/bin/python research/handoff/fetch_measured_rirs.py \
  --audit .local/mit-rir-audit.json --output "$WAKE_DATA/measured-room-rirs-v1"
```

The48-pair extraction pilot covers four samples per engine/label across six
engines. All48 unmodified control feature replays are byte-exact; treatment
keeps source/label/background/timebase pairing and changes all48 feature
arrays. Maximum response-energy difference is1.74e-7. Three targeted tests
protect input immutability, response timing/energy and identical subsequent
random draws. See
[`embedding-measured-room-pilot-v3.json`](../wakeword_p4/results/embedding-measured-room-pilot-v3.json).

The complete paired extraction and audit now pass for all 10,070 pairs. All
original control arrays match, all treatment arrays change, and the 48 original
numerical renderer replays are byte-exact. Maximum response-energy difference
is 2.61e-7. The sanitized report is
[`embedding-measured-room-feature-audit-v3.json`](../wakeword_p4/results/embedding-measured-room-feature-audit-v3.json).
Its plan binds the original local
jobs/index layout, including source paths; this is not yet a portable raw
extraction plan for a rebased public snapshot. Run on the original verified
layout, using fresh output folders:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/prepare_measured_room_views.py \
  --data-root "$WAKE_DATA" \
  --plan research/wakeword_p4/embedding/plans/measured-room-views-v1.json \
  --output "$WAKE_DATA/head-features-measured-room-v1" \
  --control-output "$WAKE_DATA/head-features-measured-room-control-v1" --workers 3
```

Use `--limit 48` only for a separately named extraction pilot; pilot completion
cannot authorize fitting. Replay the complete audit with
`embedding/audit_measured_room_views.py --data-root "$WAKE_DATA" --plan
research/wakeword_p4/embedding/plans/measured-room-views-v1.json --output
.local/fresh-room-audit.json`.

Four matched dense/temporal control/treatment plans are frozen before fitting
in `embedding/plans/measured-room-continuation-v1.json`. Each entry binds its
initial weights, training-plan hash and fresh output directory. Run the
unchanged `train_head.py` with that plan and initialization, followed by
`evaluate_head.py` with the same plan. The two arms include exactly one of the
new feature directories each; their paired IDs intentionally match, so never
load both together. Validation chooses checkpoints/thresholds before existing
paired/continuous regression. No new recognition gain, candidate acceptance
or physical trial is reported at this stage.

After all four member selections exist, derive the fixed equal means and
freeze their development protocols with the maintained helper:

```sh
.local/4090-env/bin/python research/wakeword_p4/embedding/run_measured_room_comparison.py \
  calibrate --data-root "$WAKE_DATA" --log-output .local/room-comparison-logs
```

Preserve the generated equal-mean and regression plans under `embedding/plans/`
before running the same command with `regress` instead of `calibrate`. Each
stage uses fresh output/log files, checks selected model/policy identities,
and retains the original paired/English/Mandarin protocols. These commands
perform offline inference and do not play sound or access a board.

The study is now complete and **both continuations are rejected as production
replacements**. The matched initializer and representative PTQ arrays are exact
between arms; all 50,697 training rows and 7,618,929 embedding frames per arm
were verified. Both equal means select .4 on original validation, before any
development regression. The measured treatment has 95.261% validation macro
recall versus control 95.282%, and 3 versus 2 continuous events / 7.304 h.

| Existing development measurement | Matched synthetic control | Measured-room treatment | Hi ESP |
| --- | --- | --- | --- |
| Positive clips with an event | 46 / 50 | 44 / 50 | 48 / 50 |
| Ordinary sentences with an event | 1 / 100 | 1 / 100 | 0 / 100 |
| Nyx near-word clips with an event | 26 / 90 | 25 / 90 | 0 / 90 |
| English events / 5.549 source hours | 2 | 2 | 0 |
| Mandarin events / 1.294 source hours | 0 | 0 | 0 |

The original frozen equal mean remains 46/50 with 23/90 near-word events and
only one English event. Neither new continuation establishes an improvement.
This single-seed study does not establish the cause of earlier acoustic
misses or universal effects of measured responses. No new firmware was flashed,
no new acoustic playback occurred, and final reserves remain unread. Complete
validation grids and regression exposure/confidence metrics are in
[`embedding-measured-room-validation-v3.json`](../wakeword_p4/results/embedding-measured-room-validation-v3.json)
and
[`embedding-measured-room-regression-v3.json`](../wakeword_p4/results/embedding-measured-room-regression-v3.json).

Both failed runs are preserved as research extensions, including exact feature
arrays, all 16 exported checkpoints, selected-member validation data and
portable fitting plans:

```sh
python3 research/handoff/fetch_assets.py --output .local/room-update \
  --asset measured-room-control-v1.tar.gz --asset measured-room-treatment-v1.tar.gz
```

Merge only verified files into the prior snapshot; overlapping original
control feature arrays and initial weights must be byte-identical. Raw RIRs,
microphone audio and device state are absent from these extensions. The
training feature directories share paired IDs: each training plan must load
only its corresponding arm.

Both public downloads and every archived member hash were verified. Merging
them adds10,778 files while10,072 overlapping files stay byte-identical. The
four portable fitting plans reproduce five original same-seed batches each;
both entire equal-mean calibration JSONs reproduce byte-exactly. This verifies
the handoff, not CUDA execution or wake recognition acceptance. See
[`embedding-measured-room-portable-v3.json`](../wakeword_p4/results/embedding-measured-room-portable-v3.json).

### Training supervision diagnostic and endpoint control

A frozen training-only sample covers384 distinct audio hashes across six
engines and both labels, with four canonical views per source. The original
mean detects764/768 positive variants and falsely accepts26/768 negatives.
These are in-sample diagnostics, not recognition acceptance. Uniform negative
endpoint draws intersect the source-activity interval about37–50% of the
time; that interval omits the embedding receptive field, so this does not
establish silence sampling as the cause of failures.

The first two audit prechecks stopped before scoring: identical same-role
spellings generated extra paths, and the original dataset intentionally uses
symlinks. Their exact code/plans are preserved at `39b7406` and `fd0408b`.
The corrected `training-discrimination-audit-v3.json` preserves the original384
audio hashes and canonical source IDs, explicitly binds1536 feature rows,
and checks actual metadata/role/hash identities. No opposing-label audio hash
was found among11,604 source-file rows (11,536 distinct audio hashes).

Some standalone TTS clips use the file end as the keyword marker. In the
frozen32-source OpenAI subgroup,10 markers trail the last low-energy RMS frame
by more than200ms (maximum840ms). Apparent early crossings relative to these
markers do not prove early phonetic detection. RMS endpoints are not phonetic
ground truth; the original audio and metadata remain preserved. Reports are
`embedding-training-discrimination-v3.json` and
`embedding-training-marker-diagnostic-v3.json`.

The isolated endpoint experiment changes metadata only in the added
environmental views. A standalone complete alias with a file-end marker and
more than200ms of low-energy tail receives a marker80ms after the last
10ms frame above1% of peak RMS. Source-prefix labels, negative labels, all
20,140 feature arrays and the original base views stay unchanged. The full
audit changes68 rows across34 audio hashes, all OpenAI by eligibility rather
than an engine-specific rule. Since PTQ sampling uses only original base
views, its representative inputs remain matched to the retained controls.

`embedding/plans/endpoint-views-v1.json` freezes preparation;
`embedding/plans/endpoint-continuation-v1.json` binds two fitting plans and
their retained controls. Use the unchanged `train_head.py` and original
validation-only `evaluate_head.py`. Both keep negative weight5 and all old
sampling/optimizer/selection settings. This tests only augmented-view
supervision; full corpus realignment and a separate negative-weight20
control are not combined into this experiment. No candidate is accepted.

Both endpoint continuations now complete with representative inputs byte-exact
to their retained controls. The frozen mean selects .4: source-macro validation
94.4851% versus retained control94.6845%, each with one event/7.30378h. The
existing development regression has44/50 positive clips versus historical
original mean46/50 and Hi ESP48/50. Near-word false clips20/90 versus original
23/90 improve at the expense of positive recall; ordinary1/100, English1/5.549h
and Mandarin0/1.294h stay unchanged. This candidate is rejected as a replacement.
No further acoustic playback, physical flash or reserved scoring occurred.
The matched control comparison is on validation; development references here
are the historical original mean, not a new matched-control regression.

Exact plans and results are `head-equal-mean-endpoint-v1-{plan,regression-plan}.json`,
[`embedding-endpoint-validation-v3.json`](../wakeword_p4/results/embedding-endpoint-validation-v3.json)
and [`embedding-endpoint-regression-v3.json`](../wakeword_p4/results/embedding-endpoint-regression-v3.json).
The failed candidate, feature metadata view and every fitted checkpoint are
preserved for the GPU machine:

```sh
python3 research/handoff/fetch_assets.py --output .local/endpoint-update \
  --asset endpoint-continuation-v1.tar.gz
```

This extension references unchanged environmental feature arrays already in
the original snapshot. Merge overlapping files only when byte-identical; use
the included portable training plans to bind normalized index paths. No new
waveforms or microphone/device state are exported. This remains one partial
supervision experiment, without a claim that RMS endpoints are phonetic truth
or that full-corpus realignment cannot help.

The public download and all20,497 archive members pass their recorded hashes.
After merging only identical overlaps, both portable samplers reproduce five
original batches each over40,627 rows, and the entire mean calibration JSON
replays byte-for-byte. See
[`embedding-endpoint-portable-v3.json`](../wakeword_p4/results/embedding-endpoint-portable-v3.json).

### Negative cost control and alternative model feasibility

`embedding/plans/negative20-continuation-v1.json` freezes a separate loss-cost
intervention: negative BCE weight20 instead of5, with both original control
datasets, labels, initialization, seeds, optimizers, sampling and PTQ unchanged.
Each head's100 prefit batches match its retained control byte-for-byte; actual
representatives also match. All four checkpoints per head complete training
and original validation. The selected mean has94.5595% source-macro recall
versus retained control94.6845%, two continuous events versus one/7.30378h,
and max confusing-source14.5833% versus13.9286%. Its threshold changes from.4
to.2; a calibration shift is not a discrimination gain. Candidate rejected
before development regression or physical flashing. The prepared regression
plan is unexecuted; independent final fixtures remain unread.

See [`embedding-negative20-validation-v3.json`](../wakeword_p4/results/embedding-negative20-validation-v3.json).
The `negative20-continuation-v1.tar.gz` extension preserves all checkpoints
and portable plans while reusing the original `head-features-environment-v2`
arrays. Reproduce fitting with unchanged `train_head.py` and `evaluate_head.py`
using each manifest member's explicit plan, initialization and output path;
then use the frozen `head-equal-mean-negative20-v1-plan.json` with
`evaluate_equal_mean.py`. Do not combine this failed cost experiment with
the preceding endpoint treatment or infer that all negative weighting fails.
The public archive passes all member hashes; both portable samplers reproduce
five original batches per head and the complete mean calibration JSON is
byte-identical. See
[`embedding-negative20-portable-v3.json`](../wakeword_p4/results/embedding-negative20-portable-v3.json).

MultiNet7 English is another source-reviewed option, rather than a trained
Nyx recognizer. Pinned ESP-SR2.6.1 supports custom phoneme commands on P4,
but its local model files total2,751,129bytes; individually zlib9-compressed
payloads total2,160,910bytes, exceeding the entire current2MiB filesystem.
Existing OTA app slots are also nearly full. This is not an actual packed/link
size, and it does not prove that every storage/size solution is impossible.
It does rule out treating those weights as a drop-in current-layout change.
Official timing/memory figures are advisory; no MultiNet create/detect or
keyword/near-word/resource test had run at that source-review snapshot. Preserve dual OTA, user GUI
assets, single audio producer and immediate-request FIFO before considering
any integration. See
[`multinet-source-feasibility-v1.json`](../wakeword_p4/results/multinet-source-feasibility-v1.json).

The subsequent **silent standalone P4** probe now actually initializes that
model and completes 48 cases / 7,272 detect calls. Both fixed grammars detect
zero of 12 target sources at the frozen .9 default; one positive triggers an
unrecorded non-target command in the contrastive grammar. Neither setting is
usable. There was no known-supported-command positive control, so this does
not rule out the entire MultiNet family. No microphone or speaker was opened.
No threshold/grammar was fitted to these misses, and no final audio was read.

Two model-creation failures preceded successful inference. Padding file
payloads alone did not resolve them; a retained null-path vocabulary lookup
led to a read-only RAM VFS exposing the original model/initialization files.
The original development app, service and Hi ESP listener were restored after
every hardware attempt. The dining-room production panel was not used.

The runner originally parsed only 46 cases because two UART timeout fragments
were written as separate lines. Both complete JSON payloads remained in the
original log. The offline audit joins only those two split markers, retains
the original hashes/error, and verifies all 48 identities, sample counts and
7,272 calls without another hardware run. Future UART reads now retain exact
bytes and buffer incomplete lines through timeouts. See
[`multinet-p4-silent-v1.json`](../wakeword_p4/results/multinet-p4-silent-v1.json)
and the [probe protocol and measured limits](../wakeword_p4/multinet_probe/README.md).

A separately frozen eight-source known-command control now completes16cases
with matching consumed-PCM CRCs, but both stock and replacement grammars miss
all4known commands at.9. Raw decoder strings partially match the expected
phonemes. Thus Nyx phrase quality remains uninterpretable; do not respond by
tuning thresholds on these misses. See
[`multinet-control-v1-p4.json`](../wakeword_p4/results/multinet-control-v1-p4.json).
The same pinned upstream unit test contains an original joke recording and
asserts recognition. Its subsequent P4 reference control passes2/2stock-grammar
placements at the upstream-declared.1threshold, while automatic mixed G2P
still misses2/2. Its article phoneme differs from the original stock spelling.
The separately frozen reference-v3 corrects only that spelling through the
official phoneme API and restores both mixed-grammar detections, while the
stock native events reproduce exactly. The paired audit verifies identical
model, PCM and operating point plus that single actual active-command change.
See the [reference control protocol and evidence](../wakeword_p4/multinet_probe/README.md).
No acoustic playback was performed, and no Nyx quality candidate is accepted.

## Board and numerical constraints

Read [`../panel_wake_runtime/README.md`](../panel_wake_runtime/README.md) before
building/flashing. Runtime code is opt-in through CMake and does not switch the
production default. Preserve the single I2S producer, existing AEC/AGC call
path, ordered preroll/FIFO, 16 kHz capture and 24 kbps / 20 ms Opus packets.
The experimental idle detector alone uses raw mic samples, because a matched
three-stage capture localized the old missed cue to communication AEC.

The final embedding runtime uses a bounded asynchronous consumer, measured IRAM
placement and exact padding/LUT rewrites. It destroys/joins the idle classifier
before starting a call to recover internal task-stack space, then recreates it
when rearmed. This fixes the observed call allocation failure; it does not fix
the two missed physical wakes. Shared optimized FC dispatch/rounding mismatches
are handled by the local reference-compatible evaluator. Do not substitute
XNNPACK goldens: host extraction and numeric comparisons use BUILTIN_REF.

Only the USB development panel participated. Do not copy this machine's board
addresses, USB port, account/session, keys or NVS to another environment. Select
and verify the destination development board and save its own restore image.
The runners flash only a verified active app slot and restore it in `finally`;
never write the standalone probe's partition table over a provisioned panel.

## Experiment source archive

`historical-experiment-source.tar.gz` preserves the original per-run Python and
P4 source from the output workspaces, including the earlier backbone
reconstruction/quantization, numerical probes and acoustic harnesses. These are
historical source snapshots, with original machine-specific paths and launch
dependencies; use maintained helpers under `research/` for new runs. Private
capture headers and all room recordings/device-state outputs are excluded.
The archived upstream source retains its own notices. Provenance and licensing
references are recorded in the training and embedding research READMEs.

To export a future snapshot, run `export_assets.py --help`; every output must
use a fresh folder. Publish research assets as a prerelease, never as the latest
production firmware/OTA release.

The later Mac quarter-TTS activity study is an additional, separate archive:
[`embedding-negative-activity-v2.tar.gz`](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/embedding-negative-activity-v2.tar.gz).
Its [asset manifest](assets-negative-activity-v2.json) binds the archive hash,
size and source identities. It contains the exact training-source energy
audit, v1 failure evidence, v2 interval policy/manifest, paired fitted models,
representatives/numeric fixtures, complete validation scores and event traces.
It adds no captured room audio, board state, credentials or source waveform
transformations. Source preparation is `67c09b0`, frozen fitting is `4b9bded`;
the historical failed preparation uses `6bf333f`. It reuses original feature
assets and retained initialization, with unchanged source hashes. Preserve
those historical sources rather than updating frozen helper hashes silently.
The mixed joint/confidence-only proxy result is described in
[`embedding/README.md`](../wakeword_p4/embedding/README.md); it is not Hi ESP
parity or a production firmware release. All110member hashes and GitHub's
archive digest are checked before publishing the reference.

The [phone-fit supplement](assets-phone-fit-v1.json) adds the retained
auxiliary head from the exact original-recipe replay and the frozen-encoder
head-fit diagnostic. It depends on the bank/initializer in the
[original phone-supervision handoff](assets-phone-distillation-v1.json).
Download the supplement from its manifest URL and verify the SHA256 and
`CONTENTS-microwake-phone-fit-v1.json` member hashes before extraction.
Its root is `workspace/home-wake-research-20261110/phone-fit-handoff-v1`.

It contains both auxiliary checkpoints, source/input identities, plans,
results and executed source snapshots. The original INT8 replay is exact;
the faster head-only optimizer still fails the nonblank training-fit gate.
The first joint attempt entered float validation before interruption and
produced no summary or INT8 model; its source and log are retained explicitly.
Use `--phone-warmup-only --steps 0` with the head-fit plan to reproduce the
head/cache without joint training or validation. Four numerical/source tests
pass, and the unchanged original comparison is reproduced without rescoring.
The archive download and all listed member hashes are verified. This is an
unaccepted research supplement, with no captured room audio, board state or
firmware. It is not a production release or evidence of Hi ESP parity.

The [conditional phone-study handoff](assets-phone-conditional-v1.json)
adds the subsequent .01 and .1 auxiliary-gain experiments. Excluding CTC
blank from the phone-identity objective passes the training-fit prescreen,
but both wake models still fail the frozen development gate. The archive
retains exact binary/auxiliary checkpoints, INT8 models, launched source
snapshots, validation scores/event traces, gradient and spike-retention
diagnostics, and frozen plans. The control model is byte-identical to the
previous original-recipe control; its calibration is reused without
rescoring. Both source-bank and initializer dependencies are in
`assets-phone-distillation-v1.json`.

Download from the manifest URL, verify its archive hash, and check the
`CONTENTS-microwake-phone-conditional-v1.json` member identities. Extract
into a fresh staging directory; the sibling study roots are
`workspace/home-wake-research-20261110/microwake-phone-conditional-v1` and
`microwake-phone-conditional-gain-v1`. Supporting plans/results/source are
under the same parent's `phone-conditional-handoff`. The public download and
all listed member hashes are verified. No room audio, device state or
firmware is included, and no board, microphone or speaker was used. This is
unaccepted research, not a production update.

The [full-resolution phone-bank handoff](assets-phone-sequence-bank-v1.json)
preserves all original 168 teacher outputs for each of the same 236 training
windows. Existing features and averaged probabilities remain byte-exact;
source identities and binary wake labels are unchanged. Its root is
`workspace/home-wake-research-20261110/microwake-phone-sequence-v1/bank`.
Supporting source, plans and audits are in the sibling
`phone-sequence-bank-handoff` directory. The prior phone-distillation asset
supplies the initializer; use the original feature-index identities.

The original full-resolution paths have 1,364 phone labels in 236 nonempty
strings. They are weak teacher predictions, not human-reviewed transcripts.
The archive does not claim that CTC fitting or wake quality passes. Verify
the manifest archive hash and `CONTENTS-microwake-phone-sequence-bank-v1.json`
member identities before extraction into fresh staging. The actual public
download and all listed hashes are verified. No waveforms, microphone audio,
device state, credentials or firmware are included.

The [CTC readout fit supplement](assets-phone-ctc-fit-v1.json) contains the
1,000-step and 5,000-step frozen-head checkpoints, raw metadata, launched
sources, plans, fit reports and full greedy/beam readout audit. Both runs
fail the fixed prescreen; no joint model, wake validation or firmware
candidate was produced. The longer run reproduces the original head weights,
draw digest and fit at step 1,000. Binary weights and optimizer are unchanged.
Its nine core tests and separate exhaustive beam-remapping test pass.

The auxiliary H5 files contain weights, not warmup Adam slots. Reproduce the
exact optimization from the original plan/seed rather than claiming an
optimizer resume from H5. Use the full-resolution bank dependency above,
and verify the archive hash and `CONTENTS-microwake-phone-ctc-fit-v1.json`
member identities before extracting into fresh staging. Both study roots
and the supporting `phone-ctc-fit-handoff` directory are under
`workspace/home-wake-research-20261110`. The public download and all listed
member hashes are verified. This remains unaccepted research with no source
audio, device state, credentials or firmware.

The [joint-encoder CTC supplement](assets-phone-ctc-joint-v1.json) contains
the subsequent separately planned matched study: the same 5,000-step frozen
head warmup, then 3,000 joint steps at binary learning rate .0002 with auxiliary
gain zero or .1. It retains both binary/auxiliary checkpoints, scalar INT8
exports, launched source snapshots, launch/input audits, all development
scores/event traces, plans and the paired weak-sequence diagnostic.
The archive's study root is
`workspace/home-wake-research-20261110/microwake-phone-ctc-joint-v1`;
supporting records are in the sibling `phone-ctc-joint-handoff` directory.

The wake-quality gate fails: fixed-reference macro proxy recall changes from
81.86% to 84.48%, but false near-word views rise from 74 to 87/3,744.
Both have one continuous event/7.324 hours and no zero-event grid point.
Improved auxiliary transcription does not qualify the wake model. The study
does not change the previous failed prescreens or authorize firmware use.

Use the full-resolution bank and original initializer dependencies listed in
the manifest. Full mixed-source retraining also needs the original scalar
feature tree with its exact feature-index identities; this supplement does
not include that complete feature cache. Verify the archive SHA256 and all
`CONTENTS-microwake-phone-ctc-joint-v1.json` member hashes before extraction
into fresh staging. The actual public download and all 262 listed member
hashes are verified. No source audio, room recording, device configuration,
credential or firmware is included. This is unaccepted research.

The [bounded-encoder CTC supplement](assets-phone-ctc-balanced-v1.json)
adds the separately frozen .25 auxiliary/binary encoder gradient norm cap,
with the same 5,000-step warmup and 3,000-step joint recipe. Its study root is
`workspace/home-wake-research-20261110/microwake-phone-ctc-balanced-v1`;
supporting plans, result reports, the full mixed-batch gradient diagnostic and
input auditor are in the sibling `phone-ctc-balanced-handoff` directory.
Executed sources, binary/auxiliary checkpoints, scalar INT8 exports, commands,
raw scores/event traces and silent postprocessing scripts are retained.

This recipe also fails. At .97, macro proxy recall rises from 81.86% to
83.23%, while false near-word views rise from 74 to 83/3,744. Both models
retain one continuous event/7.324 hours and no zero-event grid point. The
cap is active on 2,343/3,000 joint steps and meets its observed norm bound.
The original control export, calibration JSON and full validation-details
JSON replay byte-exactly. Improved weak-phone fit is not wake acceptance.

Use the same full-resolution bank and initializer dependencies as above.
The original complete scalar feature cache is still a separate requirement;
it is not in this supplement. H5 files store weights, not Adam slots. Verify
the manifest archive SHA256 and `CONTENTS-microwake-phone-ctc-balanced-v1.json`
member hashes before extraction into fresh staging. The public download and
all 279 listed member hashes are verified. No source waveform, room recording,
device state, credential or firmware is included. The study ran silently and
did not access a speaker, microphone or board. It remains unaccepted research.

The [compact phrase readout supplement](assets-compact-phrase-ctc-readout-v1.json)
retains the fixed 1,000-update temporal-mean and six-token CTC readouts,
executed sources, plans, launch/log, full fitted row scores and input/result
audit. Both use the original frozen encoder and the same 32,000 row draws.
CTC fails its fit gate (93.60% macro, 7.20% greedy exact sequences, normalized
loss 1.00034); no held-engine fold, open-set validation or firmware follows.
These fitted four-phrase scores are not false-wake or generalization evidence.

The study root is
`workspace/home-wake-research-20261110/compact-phrase-ctc-readout-v1`.
Its bank and initializer dependencies are listed in the manifest. The complete
scalar feature cache remains separate. H5 files retain weights, not Adam slots.
Verify the archive SHA256 and all
`CONTENTS-compact-phrase-ctc-readout-v1.json` member hashes before extraction.
The unauthenticated public download and all 85 listed hashes are verified.
The audit runtime and training-environment versions observed after the run are
explicitly distinguished. This is rejected research, with no audio, board
state, credentials or firmware, and all processing was silent.

The [compact encoder-fit supplement](assets-compact-phrase-ctc-encoder-fit-v1.json)
contains the separately frozen representation-learning diagnostic. The full
frozen-head reference replays exactly; training the encoder and head for the
same 1,000 updates fits all 236 training rows, including greedy two-token
order. This qualifies only the design of a later mixed-source wake study.
It is not independent recall, false-wake or deployment evidence.

The root is
`workspace/home-wake-research-20261110/compact-phrase-ctc-encoder-fit-v1`.
Executed sources, plans, final encoder/head/frozen-reference weights, raw row
scores, log, launch and input/result audit are retained. Use the manifest's
bank/initializer dependencies; the full scalar feature cache is separate.
H5 weights do not retain Adam slots. The unauthenticated download and all 88
listed member hashes are verified. No audio, device state or firmware is
included, and the complete Hi ESP comparison remains open.

The [full compact-sequence supplement](assets-compact-phrase-ctc-v1.json)
retains the subsequent binary-only versus compact CTC auxiliary experiment.
Both share 1,000 frozen-head warmup and 3,000 mixed-source joint updates.
The original binary control export and input/PTQ identities replay exactly;
its prior development score files are reused byte-exact after that audit.
The new treatment's full validation and comparison finish successfully, but
the recipe fails the fixed advancement gate: macro proxy recall improves
at .97 while false near-word views increase, and one continuous event remains.
The fitted sequence head is retained for research and discarded from both
scalar exports. It does not qualify the detector for deployment.

The study root is
`workspace/home-wake-research-20261110/microwake-compact-phrase-ctc-v1`.
Executed sources, launch/plan, warmup and final binary/auxiliary weights,
scalar INT8 models, all development scores/event arrays, read-only input audit,
changed-plan rejection check and cached error-change analysis are included.
Supporting postprocessing scripts and their provenance are retained under
`repo/research`. Bank/initializer dependencies are explicit, and the full
scalar feature cache is separate. H5 files do not contain Adam slots.
The unauthenticated public download and all 279 listed member hashes are
verified. No source audio, device state or firmware is included. Processing
stayed silent, and real-speaker, latency, resource and call parity remain open.

The [retained-sequence inference supplement](assets-compact-sequence-inference-v1.json)
contains the failed first export procedure, corrected numerical smoke and
completed matched binary versus absolute CTC-product development experiment.
The corrected export passes all 2,603 aligned training windows and exact
quantized reset/chunking replays. The fixed inference rule is rejected:
at its best strict 1%/zero-continuous-event point, macro proxy recall is only
1.53125%. Numerical export correctness does not establish wake quality.

The roots are `workspace/home-wake-research-20261110/compact-sequence-export-smoke-v1`,
`compact-sequence-export-smoke-v2` and `compact-sequence-inference-v1`.
Executed sources, immutable plans/launches, original failure log, dual INT8
model, complete development score traces and read-only reconstruction audit
are retained. The audit reconstructs all 6,136 clip maxima/timebases,
35 continuous sources, both event streams, grids and the rejection decision.
The full scalar feature cache remains external; its index hash alone does
not verify every feature byte. H5 checkpoints in dependencies omit Adam slots.

The unauthenticated public download, archive SHA256 and all 342 listed member
hashes are verified against `CONTENTS-compact-sequence-inference-v1.json`.
Use the two dependencies in the manifest. SavedModel directories are omitted;
the actual INT8 model is retained. No source audio, device state, credentials
or firmware is included. The full-history sequence output is an unoptimized
research interface with unmeasured P4 costs, and is incompatible with the
current scalar adapter. No playback, microphone or board operation occurred.

The [uncapped compact training bank](assets-compact-broad-bank-v1.json)
retains all eligible existing same-voice/render Cartesian pairings instead of
the earlier eight-pair source cap. It has 5,060 pairs / 10,120 retained rows,
representing 3,112 distinct midpoint crops from 744 original source hashes
(404 positive, 340 negative). Pairs and four environment renders are
correlated. Uniform pair sampling changes engine/phrase exposure; the bank
is intended for a new matched control/treatment study, with no inherited
wake-quality claim.

The root is `workspace/home-wake-research-20261110/compact-broad-bank-v1`.
The archive includes the NPZ feature bank, per-row source/crop identities,
immutable preparation source/plan/launch/log, original feature index and
whole-phrase window bounds. The full 2,984,015,040-byte scalar feature cache
is external, bound separately by SHA256. No teacher output or waveform is
included. It uses unchanged original labels, render inputs and PCM speed.
The actual unauthenticated public download and all 95 listed member hashes
pass against `CONTENTS-compact-broad-bank-v1.json`.

The subsequent separately frozen broader-bank study has completed and is
rejected. This bank publication reports data preparation only. Both arms
rebuilt binary-only and gain-.1 CTC models on identical new inputs and the
original fixed budget; no previous control scores were reused. Processing
stayed silent and did not access a board. Real-speaker and full-device Hi ESP
acceptance remain open.

The [complete broader-bank study supplement](assets-compact-broad-ctc-v1.json)
retains both fixed final scalar INT8 models and binary/auxiliary checkpoints,
the original initializer, immutable source/plan/launches, runtime versions,
warmup logs, complete development score traces, input audit and supplementary
cached-score reconstruction. At .97, macro proxy recall rises from 78.2569%
to 81.1458%, while near-word false views rise from 67 to 76/3,744 and continuous
events rise from two to three/7.323822 hours. Neither arm has a strict
1%/zero-event point in the frozen grid. There is no qualified deployment
candidate, and no final/test score or threshold extension follows.

The root is `workspace/home-wake-research-20261110/microwake-compact-broad-ctc-v1`.
Use the broader-bank dependency above; the full scalar feature cache remains
external. Both exports discard the auxiliary head and remain 61,712 bytes /
25,831 parameters. H5 files omit Adam slots. The first supplementary audit
launch used system Python without LiteRT; its failed launch/log are retained
alongside the successful rerun in the pinned training Python. No source,
scores or tolerance changed. Reconstruction checks each arm's 6,136 clips /
1,429,512 smoothed steps, 35 continuous sources / 878,709 steps, all maxima,
timebases, independent event streams, full grids and the rejection decision.
The unauthenticated download and all 303 listed member hashes are verified
against `CONTENTS-microwake-compact-broad-ctc-v1.json`. No waveform, device
state, credentials or firmware is included; SavedModel directories are omitted.

To rerun the supplementary accounting after restoring this study and its
original external feature tree, choose a fresh output filename:

```sh
PYTHONHASHSEED=0 "$WAKE_PYTHON" research/handoff/audit_scalar_wake_scores.py \
  --artifacts "$WAKE_OUTPUT/microwake-compact-broad-ctc-v1" \
  --features "$WAKE_FEATURES" \
  --output "$WAKE_OUTPUT/microwake-compact-broad-ctc-v1/rechecked-scores.json"
```

This command checks saved accounting without replaying model invocations.
The full comparison and input-audit commands are retained in the study's
launch files. No speaker, microphone or board operation is required.

The [same-weight scalar precision supplement](assets-scalar-precision-diagnostic-v1.json)
contains the training-only float export smoke and subsequent complete
float/INT8 development diagnostic. Both float TFLite exports reproduce the
original saved streaming graphs and full windows within 4.86e-6; float reset
and 256-step chunking replay exactly. The new INT8 adapter inherits the
original path unchanged. These checks validate the arithmetic comparison,
not wake quality or P4 suitability.

Neither original model has a strict point in float or INT8 within the frozen
21-point grid. At .97, control float/INT8 macro proxy recalls are
79.4013%/78.2569%, with 70/67 false near-word views and two continuous events
each; compact CTC recalls are 79.2009%/81.1458%, with 73/76 false near-word
views and three events each. Exposure is the same 7.323822 development hours.
The result rules out a precision-only explanation for the absent strict point
under these weights/inputs/grid. It does not show that quantization is harmless:
large local score differences remain, and precision both adds and removes
clip detections. Equal event counts do not prove equal event locations.

The roots are `workspace/home-wake-research-20261110/scalar-precision-export-smoke-v1`
and `scalar-precision-diagnostic-v1`. The archive retains immutable sources,
plans, launches/runtime versions, both float models, complete paired clip
scores, both precisions' continuous traces, original INT8 models/H5 weights,
and the original scalar SavedModels needed for export reconstruction. Original
SavedModels supplement the dependency archive, which omitted them. H5 files
alone do not provide optimizer-resume checkpoints. Exact original commands
and paths remain provenance; a new host should record its own fresh launch
with its relocated inputs rather than edit those historical records.

Fresh INT8 inference replays every original clip/continuous score exactly.
The read-only auditor binds committed plans and actual commands, source/model
bytes, parent evidence, the full scalar binary and all 35 actual continuous
feature files. It reconstructs all maxima, timestamps, independent events,
full grids, strict-point existence, contingencies and error aggregates.
The existing broad-study archive is the explicit dependency. The full scalar
and continuous feature tree remains external; index hashes alone cannot
replace those byte checks. No final/test inference or deployment follows.

The unauthenticated public download, archive SHA256 and all 356 listed member
lengths/hashes pass against `CONTENTS-scalar-precision-diagnostic-v1.json`.
No source waveform, device state, credential or firmware is included. Work
was silent and used no microphone or board. Physical wake latency, P4 costs,
real-speaker performance and voice-call stability remain unproven.

The [explicit ambient compact-target study](../wakeword_p4/results/microwake-background-ctc-v1.json)
keeps compact CTC gain .1 in both arms and adds per-frame empty-target CTC
loss on the existing 48 ambient-negative slots at gain zero versus one.
The marker travels with each sample through the shuffle; ordinary TTS
negatives retain their existing role. Both models are retrained with the
same 5,060-pair bank, 1,000-step frozen-head warmup and 3,000 joint updates.
Each arm observes 144,000 marked background examples. Test arrays are unopened.

The zero-gain control exactly replays the parent compact-CTC INT8 model,
legacy first 16 batches, full warmup, pair schedule and PTQ input. Its original
development cache is reused only after replay and byte-bound input checks.
All 347 training and 35 development background NPY containers and canonical
uint16 payloads are hash-bound in `background-inputs.json`, along with the
original scalar binary and bank. Exact-array and metadata disjointness are
checked; these do not establish semantic labels or absence of partial overlap.

The result is rejected: at .97, macro proxy recall changes 81.1458% → 79.0650%,
continuous events 3 → 1 over 7.323822 hours, and false near-word views 76 → 77.
Worst-group false fraction remains 4.1667%; neither model has a strict point
in the original grid. No firmware or inference rule is changed.

To reproduce the independent accounting with a fresh output filename:

```sh
"$WAKE_PYTHON" research/handoff/audit_scalar_wake_scores.py \
  --artifacts "$WAKE_OUTPUT/microwake-background-ctc-v1" \
  --features "$WAKE_FEATURES" \
  --output "$WAKE_OUTPUT/microwake-background-ctc-v1/rechecked-scores.json"
```

The original training and finishing commands, runtime versions and immutable
sources are in `study-launch.json`, model `launch.json` and `postprocessing/`.
The supplementary audit reconstructs complete traces and the gate. A changed
plan is rejected before input/scoring access. All processing stayed silent;
no microphone, speaker, board, production device or final/test inference ran.

The [complete background-study archive](assets-microwake-background-ctc-v1.json)
uses root `workspace/home-wake-research-20261110/microwake-background-ctc-v1`.
It retains both original scalar SavedModels, H5/INT8 models, initializer,
immutable source/plan/launch/runtime records, background-byte inventory,
control-cache provenance, full development traces and all audits. The parent
broad-study and broader-bank archives are explicit dependencies. The complete
scalar/background feature tree is external; historical launch paths are
provenance and must not be rewritten in place on a relocated machine.

The public unauthenticated download verifies the exact 325-member inventory
and all 324 listed lengths/hashes. No source audio, device state, credential
or firmware is included. H5 files are weight checkpoints without Adam slots.

The [retained background-head export smoke](../wakeword_p4/results/compact-background-export-smoke-v1.json)
and [separate fixed inference comparison](../wakeword_p4/results/compact-background-inference-v1.json)
test the temporal head discarded by that scalar-only recipe. Export passes
20 predeclared training sources, float scalar/logit alignment within 1.15e-5
at the original 1e-4 gate, and exact reset/256-step chunk replay. The dual
model is 63,680 bytes / 26,197 parameters. PTQ uses 100 frozen training-bank
row IDs representing 96 distinct feature crops. No fitting or validation
feature is used in the export smoke.

The subsequent development result is rejected. Absolute CTC product reaches
the strict false-wake gate at .95 with zero continuous events and .8929%
worst-corpus fraction, but only 20.3110% macro proxy recall (8.1522% aggregate
variant recall). The matched same-artifact binary comparator has 74.6706%
macro recall at its minimum-false zero-event .99 fallback. At fixed .97,
binary/product macro recalls are 80.3061% / 10.3824%, false near-word views
71 / 6, and continuous events two / zero over 7.323822 development hours.
New dual PTQ changes binary outputs relative to the scalar-only model;
this comparison isolates the fixed inference rule within the dual artifact.

`input-result-audit.json` binds the committed/executed plan and source, actual
commands, model, scalar bytes and all training/development background payloads.
It reconstructs all clip maxima/timebases, both independent event streams,
full grids, product dominance and the rejection. The generic read-only auditor
also reproduces the old compact study; an added threshold is rejected before
input/score access. To rerun with a fresh output name:

```sh
"$WAKE_PYTHON" research/handoff/audit_compact_inference.py \
  --artifacts "$WAKE_OUTPUT/compact-background-inference-v1" \
  --repository "$WAKE_REPOSITORY" \
  --export "$WAKE_OUTPUT/compact-background-export-smoke-v1/export" \
  --features "$WAKE_FEATURES" \
  --plan-repository-path research/wakeword_p4/training/plans/compact-background-inference-v1.json \
  --output "$WAKE_OUTPUT/compact-background-inference-v1/rechecked.json"
```

No score normalization, threshold extension, final/test inference or hardware
change follows. This is a rejected fixed recipe; it establishes no real-speaker,
physical latency, resource or call stability parity with Hi ESP.

The [complete retained-head supplement](assets-compact-background-inference-v1.json)
contains both roots `workspace/home-wake-research-20261110/compact-background-export-smoke-v1`
and `compact-background-inference-v1`. It includes the dual SavedModel/INT8,
immutable plans/sources/launches/runtime, numerical diagnostics, full paired
development scores and independent audits. Its explicit dependency is the
completed background-training archive; full features remain external and
byte-bound. A fresh unauthenticated download verifies all 279 listed member
lengths/hashes and the exact 280-member inventory. No waveform, device state,
credential or firmware is included. All jobs are terminal; work stayed silent.

The [keyword occurrence diagnostic](../wakeword_p4/results/keyword-occurrence-training-diagnostic-v1.json)
separates exact whole-window CTC from the probability that adjacent collapsed
tokens `[hey,nyx]` occur inside a longer sequence. A three-state log recurrence
retains blank/repeat semantics and allows surrounding symbols. Exhaustive
short-path tests and an independent probability-domain matrix reconstruction
verify the event calculation. This is not a calibrated wake confidence: the
uninformative67-step six-class prior already has.9024687 occurrence probability.

The fixed40-source optimization-seen diagnostic retains8,513 scalar/logit
steps. At fixed.97, isolated Nyx detections change5/2/4 for scalar/exact/occurrence;
Piper continuation detections change16/0/6. At the scalar peak only2/20
continuations contain the greedy keyword, both exact, so context semantics
alone does not explain the gap. Cohort/engine/wording are confounded. Separately,
all1,702 distinct Piper training continuation sources are absent from the
compact paired bank; the encoder sees them via BCE, but they receive no paired
temporal CTC target. This motivates a new training study, not a proven cause.

The [first inference protocol](../wakeword_p4/results/keyword-occurrence-inference-v1-protocol-failure.json)
stops before development inference: the40 sources are at most252 steps, so
the required256-step cross-chunk check has no eligible source. The
[corrected, separately frozen protocol](../wakeword_p4/results/keyword-occurrence-inference-v2.json)
adds one predeclared823-frame training source; all41 scalar/product replays
pass, and the long source's full logits also replay across the boundary.
No tolerance, model, PTQ, grammar, validation grid or quality gate is relaxed.

Full development inference is rejected. At.97, occurrence-product macro proxy
recall is52.1939%, with20 false near-word views, zero continuous events and
2.1429% worst-corpus fraction. Its best strict point is.994 /17.9211% macro
recall, far below the same-artifact binary fallback at.99 /74.6706%. Every
fresh binary clip/continuous score replays the preceding dual-model study
exactly. The read-only audit binds source/plan/model/full feature bytes and
reconstructs all traces, grids, independent events and the rejection. No
threshold extension, fitting, final/test score, sound, recording or board action
occurred. Original executed commands and both failed/corrected protocols remain
provenance; relocated machines should record fresh launches.

The [complete occurrence supplement](assets-keyword-occurrence-inference-v2.json)
retains three roots under `workspace/home-wake-research-20261110/`:
`keyword-occurrence-training-diagnostic-v1`, the failed numerical protocol
`keyword-occurrence-inference-v1`, and completed `keyword-occurrence-inference-v2`.
The previous retained-head archive supplies the identical dual model; complete
feature caches remain external. Actual logits, independent probability audit,
41-source adapter replay, full development traces and original commands/source
identities are included. Its fresh public unauthenticated download verifies
all426 listed member lengths/hashes and the exact427-member inventory. No
source waveform, device state, credential or firmware is included.

### Ordinary-positive compact CTC coverage experiment

The [matched plan](../wakeword_p4/training/plans/microwake-positive-ctc-v1.json)
adds compact `[hey,nyx]` CTC to existing complete ordinary-positive crops,
keeping paired and ambient losses, all draws, crops, augmentation, warmup and
PTQ inputs fixed. Gain 0 exactly replays the parent background-treatment
scalar model and historical prefixes; gain 1 is the treatment. The fourth
label marker follows shuffle. Actual per-step crops and eligibility are
retained and independently reconstructed: 72,000 eligible draws per arm,
including 6,760 Piper continuation draws, from 3,603 source hashes.

The [completed result](../wakeword_p4/results/microwake-positive-ctc-v1.json)
is rejected: fixed .97 macro recall 79.0650% -> 79.6334%, false near-word views
77 -> 75 /3,744, continuous events 1 -> 3 /7.323822 h. No strict point exists
in either frozen grid. Sixteen software checks, three adversarial metadata
rejections and complete score-accounting reconstruction passed. Software
correctness does not establish wake quality; no head-based inference use or
firmware acceptance is implied by this scalar-only result.

The [public archive](assets-microwake-positive-ctc-v1.json) retains the complete
run at `workspace/home-wake-research-20261110/microwake-positive-ctc-v1`:
both immutable training source copies, checkpoints, SavedModels, scalar INT8,
positive exposure traces, actual launches/logs/runtime, full development score
traces and all audit code/results. The 341-member inventory and all 340 listed
hashes/lengths were checked after a fresh unauthenticated download. Dependencies
are the prior background study and broad bank archives; complete scalar and
background features remain external and byte-bound. No credentials, device
state, source waveforms or deployable firmware are included.

Use [launch_positive_ctc_study.py](launch_positive_ctc_study.py) for a fresh
reproduction after restoring the original feature/initializer/bank inputs and
pinned runtime. Required arguments are `--artifacts` (fresh directory),
`--repository`, `--upstream`, `--features`, `--bank`, `--initial-weights`,
`--prior-study` (parent background run) and `--background-inputs` (inventory).
The launcher checks committed source/plan identities, copies private source
snapshots, runs both arms and stops on a failed parent replay. Historical
absolute launches remain provenance; relocated experiments need fresh launches.
For existing local evidence, input audit uses `audit_compact_study.py` with
`--plan-repository-path research/wakeword_p4/training/plans/microwake-positive-ctc-v1.json`;
the supplementary `audit_scalar_wake_scores.py` reconstructs the saved scores.

### Retained ordinary-positive head: numerical pass, quality rejection

The [separate export plan](../wakeword_p4/training/plans/compact-positive-export-smoke-v1.json)
and [numerical result](../wakeword_p4/results/compact-positive-export-smoke-v1.json)
retain the rejected treatment's temporal head. All 20 fixed training sources
pass float window/stream alignment and INT8 reset/chunk checks. The original
training-only calibration input digest replays exactly. The 63,680-byte dual
model is a reference implementation with unmeasured P4 cost and latency.

The [separately frozen occurrence plan](../wakeword_p4/training/plans/compact-positive-occurrence-v1.json)
first verifies 41 training sources /8,787 raw steps, including the actual long
source across 256 steps. Monolithic versus chunked scalar, all logits and
products match exactly; independent probability-domain reconstruction differs
by at most 3.21965e-15. Every new-model trace is retained. The [complete result](../wakeword_p4/results/compact-positive-occurrence-v1.json)
is rejected: at .97, macro recall 78.3631% binary versus 58.8254% product,
false near-word views 71 versus 28 and continuous events 2 versus 0. The best
strict product point .994 reaches only 31.6131% macro recall (9 false views,
zero continuous events), below the frozen binary comparator's 69.4876%.
The independent score auditor reconstructs all clips, streams and the gate.

The [public supplement](assets-compact-positive-inference-v1.json) includes both
`compact-positive-export-smoke-v1` and `compact-positive-occurrence-v1` roots
under `workspace/home-wake-research-20261110/`. It retains SavedModel, dual
INT8, actual training traces, complete development score traces, frozen source,
plans, launches, logs and audit code/results. Parent ordinary-positive training
and broad-bank archives provide dependencies. Full features remain external.
No source WAV, credential, device state or firmware is included.

For a fresh run, use [launch_positive_inference.py](launch_positive_inference.py)
with `--artifacts` (new directory), `--repository`, `--features`, `--export`
(validated new dual export) and `--training-study` (positive-supervision run).
It validates actual bytes and stops before development inference if the
training-only adapter gate fails. For retained local results, run
`audit_compact_inference.py` with the usual artifacts/repository/export/features
arguments and `--plan-repository-path research/wakeword_p4/training/plans/compact-positive-occurrence-v1.json`.
Historical paths are provenance; relocated experiments need fresh launches.

### Shared intent-phone versus atomic word diagnostic

The [frozen plan](../wakeword_p4/training/plans/intent-phone-generalization-v1.json)
tests a distinct training-only representation hypothesis after the prior
scalar/head-filter failures. It replaces the four atomic second-word labels
with shared dictionary-like phone strings, without using ASR labels or changing
source intent. The original Cartesian bank reduces to 3,112 exact unique
feature crops. A full fitted diagnostic and five engine-held-out fine-tunes
compare both recipes at 1,000 steps each, keeping initialization and row draws
paired. Parent encoder exposure, different head sizes and true target lengths
are explicitly limited; these are not independent speaker/wake tests.

The [completed result](../wakeword_p4/results/intent-phone-generalization-v1.json)
rejects this recipe: engine-macro accuracy 81.1866% words versus 57.0649%
phones, confusers predicted Nyx 74/1,360 versus 526/1,360. All 12 final
checkpoints/scores, 3,112 actual crop identities, fold source exclusions,
draw schedules and metrics are retained and independently audited. The phone
head fails to fit complete greedy strings even on the full training bank;
there is no automatic follow-on wake training, validation or firmware action.

The [separate numerical plan](../wakeword_p4/training/plans/intent-phone-numerics-v1.json)
and [result](../wakeword_p4/results/intent-phone-numerics-v1.json) check both
frozen full-fit checkpoints on 60 predefined training crops. Compiled TF2.21
CTC loss and logit gradients closely match an independent float64 expanded-state
forward/backward reference. Actual phone loss/gradient maximum deltas are
3.98431e-6 /4.61202e-7, with no missing gradient above reference magnitude .001.
Fixed synthetic logits and exact-path/finite-difference reference tests are
also retained. This does not support attributing the fit failure to a large
CTC numerical error; it does not prove a unique optimization/data cause.

The [public evidence archive](assets-intent-phone-generalization-v1.json)
contains both `intent-phone-generalization-v1` and `intent-phone-numerics-v1`
under `workspace/home-wake-research-20261110/`: actual checkpoints, crop/fold
inventories, score files, numerical logits/losses/gradients, immutable source,
plans, launches, runtime, logs and all audits. The broad-bank and original
broad-study archives supply bank/initializer dependencies; full scalar features
remain external and byte-bound. No source audio, credential, device state or
firmware is included.

Use [launch_intent_phone_fit.py](launch_intent_phone_fit.py) for a fresh training
diagnostic with `--artifacts`, `--repository`, `--upstream`, `--features`,
`--bank` and `--initial-weights` in the pinned runtime. For existing artifacts,
[audit_intent_phone_fit.py](audit_intent_phone_fit.py) accepts those same input
paths except upstream (read from the launch), plus `--output`. The independent
numerical script [audit_intent_ctc_numerics.py](../wakeword_p4/training/audit_intent_ctc_numerics.py)
uses `--plan`, `--study` (the completed representation run), `--upstream`,
`--features`, `--bank`, `--initial-weights` and a fresh `--output`. Original
absolute launch paths remain provenance; relocated runs must record new launches.

### Matched-endpoint local/deep intent-phone fit

The [separate frozen plan](../wakeword_p4/training/plans/intent-phone-local-tap-v1.json)
and [completed result](../wakeword_p4/results/intent-phone-local-tap-v1.json)
compare a 41-feature local tap with the 137-feature final encoder at identical
67 right endpoints. Both fit the same 3,112 unique training crops with an
identical 610-parameter phone head, draw schedule and 1,000-update budget.
The deep control exactly replays the prior full-fit phone tensors/metrics.
Local average intent accuracy falls from 68.0504% to 43.2110%; neither arm
emits a complete greedy target. Both fail the unchanged fitted-bank gate.
This is neither independent recognition nor open-set false-wake evidence.
Depth/trainable counts differ, and local coverage starts at feature 96 while
deep coverage starts at 0; do not infer a unique cause or a universal limit.

The [complete evidence archive](assets-intent-phone-local-tap-v1.json) retains
both checkpoints, full logits, scores, crop inventory, plans, immutable sources,
launch/runtime/logs, original failed audit and explicit corrected audit.
Every checkpoint logit and four-class likelihood replays bitwise; an independent
float64 CTC reference checks the predefined first crop per engine/class.
The [audit correction](../wakeword_p4/training/plans/intent-phone-local-tap-audit-correction-v1.json)
only handles a 1.33e-15 mean-loss reduction difference with 1e-12 numerical
allowance; quality gates, original training artifacts and integer counts remain
unchanged. Nothing proceeds to firmware or reserved quality evaluation.

To replay the frozen training, use
[launch_intent_phone_taps.py](launch_intent_phone_taps.py) from training commit
`29ae1f2` in a separate checkout, with
`--artifacts`, `--repository`, `--upstream`, `--features`, `--bank`,
`--initial-weights` and `--parent-study` (the extracted shared-phone study root)
to start a fresh pinned-runtime fit. Its historical auditor retains the
documented roundoff failure; run the corrected auditor from this branch
afterward, recording a fresh correction JSON for that run and preserving
its original failed audit. For the original retained results,
[audit_intent_phone_taps.py](audit_intent_phone_taps.py) takes the same options
except upstream, plus `--output` and the retained `--audit-correction` JSON.
Retain original absolute launch paths as provenance; relocated runs need new
launch records. No audio, device credentials, firmware or full scalar feature
cache is included; the prior phone study, broad bank and broad-study initializer
archives are dependencies.

### Neutral initialization and conditional intent discrimination

Two separate completed fitted-bank experiments share a
[complete evidence archive](assets-intent-phone-neutral-and-conditional-v1.json).
The [neutral initialization plan](../wakeword_p4/training/plans/intent-phone-neutral-initialization-v1.json)
and [result](../wakeword_p4/results/intent-phone-neutral-initialization-v1.json)
replace only the random phone-head kernel with zero initialization. Complete
greedy targets increase from 0 to 2,107/3,112 and mean CTC NLL/token falls
3.18186 -> .205870, but near-word discrimination remains poor.

The [separate conditional plan](../wakeword_p4/training/plans/intent-phone-conditional-discrimination-v1.json)
and [result](../wakeword_p4/results/intent-phone-conditional-discrimination-v1.json)
keep that neutral initialization and add gain-one cross-entropy over the four
raw phone-string CTC likelihoods. On these same fitted training crops, mean
intent accuracy improves 69.0050% -> 92.9754%, Mix recall 28/448 -> 337/448,
and confusers assigned Nyx 542/1,360 -> 114/1,360. Complete greedy targets
reach 2,332/3,112. This still fails the unchanged fit gate; neither recipe
proceeds to held-engine scoring, wake export or hardware. These are training
fit results, without independent speaker or open-set false-wake acceptance.

Both controls replay their respective actual parent tensor hashes and metrics
exactly. The read-only audits reload all four final checkpoints and replay
every original logit/four-class likelihood bitwise, reconstruct initial states
and statistics, exact crops/draws/metrics/gates, and check frozen likelihood
samples against independent float64 CTC. Source snapshots, plans, runtime,
launches, failed/passing test logs and final arrays are retained. The invalid
first-call traced-Adam validation bug found by a software test was corrected
before fitting; its raw failure remains. TF2.21's nonfatal MLIR optimizer
messages are also visible in raw logs; numerical and update tests pass.

Use `launch_intent_phone_taps.py --plan <repository-plan-path>` from each
recorded training commit for a fresh fit with the same input options as above.
Initialization source commit is `eabd4ff`; its parent is the original shared
phone study. Conditional source commit is `b29b76b`; its parent is the neutral
initialization root. Historical plans require their recorded source versions.
For retained studies, use each executed `auditing/audit_intent_phone_taps.py`
snapshot without an audit-correction option; it resolves the study's committed
repository-plan path and verifies its own frozen source bytes. Original
absolute launch paths remain provenance; relocated runs need fresh launch
records. The original phone/broad-bank/broad-study initializer archives are
dependencies. No source audio, credentials, device state, firmware or full
scalar feature cache is included.

### Conditional word objective in full mixed-source wake training

The [word-intent plan](../wakeword_p4/training/plans/microwake-word-intent-ctc-v1.json)
and [completed result](../wakeword_p4/results/microwake-word-intent-ctc-v1.json)
test the conditional four-intent CTC objective directly in scalar wake training.
Both arms replay the original binary initializer, 1,000-step frozen-head CTC
warmup and 3,000 mixed-source updates. Existing binary BCE and word CTC gain .1
remain; treatment alone adds gain-one conditional intent loss on the same 16
paired slots per 128 examples. Warmup, ordinary/background losses, sampling,
checkpoint choice, scalar interface and development gates remain unchanged.

This recipe is rejected. At the existing .97 reference, mean-engine recall
changes 81.1458% -> 81.8125%, negative false clips 76 -> 83/3,744, and continuous
events stay 3/ 7.32382 hours. Worst-group false fraction remains 4.16667%; unlike
control, treatment has no threshold with zero continuous events on the frozen grid.
Saved-score text groups show almost unchanged Mix/Next/Nick confusion and
eight additional false clips on other negative texts. No reserved-quality
evaluation or firmware follows this failed development screen.

The [complete evidence archive](assets-microwake-word-intent-ctc-v1.json)
contains both actual scalar exports, SavedModels, all binary/auxiliary
checkpoints, exact warmup, source snapshots, commands/runtime, 6 passing
preflight tests, raw training/evaluation logs and complete development score
arrays. Exact control replay is required before its old scores are reused.
The separate read-only reconstruction reproduces all maxima/timebases/events,
full grids and the decision; fixed-reference error groups reconstruct totals.
The public download verifies all 362 listed member hashes and lengths. Source audio,
credentials, device state, firmware and full scalar/background features are
excluded. Broad-bank and original broad-word study archives are dependencies.

For a fresh run, use training commit `2a6d8e3` and
`launch_positive_ctc_study.py --plan research/wakeword_p4/training/plans/microwake-word-intent-ctc-v1.json`,
supplying `--artifacts`, `--repository`, `--upstream`, `--features`, `--bank`,
`--initial-weights`, `--prior-study` (the original broad-word study root) and
`--background-inputs`. The launcher records both immutable source commands
and runs the captured input audit, exact-control score reuse, new INT8
validation and frozen comparator. After completion, run the captured
`postprocessing/audit_scalar_wake_scores.py --artifacts <root> --features <features> --output <fresh JSON>`.
The archive's `diagnostics/run-post-training-audits.py` records this command's
actual log/exit and the secondary fixed-reference breakdown; its original
absolute feature path needs updating for a relocated fresh run. Preserve the
historical sources and launch paths as provenance; never overwrite retained
scores or claim cached reconstruction is independent inference/acceptance.

### Prepared word encoder initialization comparison

The [new plan](../wakeword_p4/training/plans/microwake-word-encoder-initialization-v1.json)
tests the retained training-only full-fit word encoder as a different initial
tensor group for the original scalar BCE plus .1 word CTC recipe. Both arms
keep the original scalar classifier at transfer, then train it normally during
the same 3,000 downstream updates. The old word head and optimizer are discarded;
both fresh auxiliary heads receive the same 1,000-step frozen-encoder warmup
recipe and row draws. Encoder-dependent caches/head fits intentionally differ.
Control must still replay its original complete warmup and scalar model exactly.
The unchanged development grid and quality gates apply. No result or accepted
model is claimed by preparing these weights.

The [prepared initializer archive](assets-word-encoder-initializer-v1.json)
contains all 28 byte-bound preparation files, original scalar initializer,
conversion/verification code, raw parent word checkpoint/report/plan/audit,
actual commands/logs and synthetic roundtrip probes. A fresh unauthenticated
download verifies all 30 listed member hashes and lengths. It contains no
audio or full feature caches. Training-only intent labels, corpus exposure,
weak held-engine generalization and retrospective initializer choice remain
limitations; no preparation probe measures wake quality.

Use training source commit `2e9277f` with the same launcher/input options as
the preceding study, changing `--plan` to
`research/wakeword_p4/training/plans/microwake-word-encoder-initialization-v1.json`
and adding `--treatment-initializer <extracted word-encoder-initializer-v1>`.
The parent control remains `microwake-compact-broad-ctc-v1`. The launcher checks
the complete prepared inventory before copying it into the new run and points
only treatment at its scalar weights. Original features, background inventory,
bank and pinned upstream/runtime remain separate dependencies. The live Mac
run and any later full results retain their own commands, raw logs and audits.

The [completed result](../wakeword_p4/results/microwake-word-encoder-initialization-v1.json)
rejects this initializer recipe. At the existing .97 reference, mean-engine
recall improves 81.1458% -> 83.4539% and continuous events decrease 3 -> 1 on
7.32382 reused development hours, while negative false clips increase
76 -> 93/3,744 and worst-group false fraction rises 4.16667% -> 5.20833%.
Treatment has no zero-event point on the frozen grid. Its minimum worst-group
fraction is still 3.75%, above the unchanged 1% gate. Fitted CTC exact targets
increase 8,860 -> 9,526/10,120 repeated rows without qualifying scalar rejection.
These observations do not establish independent generalization or a unique
cause. No reserved-quality scoring or hardware follows this failed screen.

The [full study archive](assets-microwake-word-encoder-initialization-v1.json)
includes both actual scalar exports and SavedModels, every checkpoint, the
complete prepared initializer, frozen source/plan/commands/runtime, raw
preflight/training/evaluation logs and complete development scores. The input
audit verifies both original-control replay and explicitly different initial
encoder identities. Full warmups are retained per arm; the default full-equality
rule for ordinary matched studies remains unchanged. Separate saved-score
audits reproduce every timebase, maximum, continuous event, full grid,
reference text group and decision. A fresh public download verifies all 397
listed member hashes and lengths. Full scalar/background caches remain
external and byte-bound; no source audio, credentials, firmware or device state
is included. Use the captured postprocessing/diagnostics with fresh output
paths after the launcher completes; preserve historical source/launch records.

## Training-only source-balanced word sampling

The [sampling inventory](../wakeword_p4/results/intent-source-sampling-inventory-v1.json)
measures unequal weights caused by Cartesian partners and correlated crops.
The [separate fixed plan](../wakeword_p4/training/plans/intent-source-balanced-sampling-v1.json)
compares uniform unique crops with equal engine/class/source-hash/render/crop
sampling at the original1,000-update word-CTC budget. The hierarchy uses only
the current fold's training IDs. Original loss, labels, initialization and
optimizers remain fixed; equalization changes class priors and multiple levels
of exposure together. The default old sampling path retains exact RNG behavior.

The [completed result](../wakeword_p4/results/intent-source-balanced-sampling-v1.json)
does not advance. All six controls exactly replay previous final tensors,
scores, decodes, metrics and draws. All12 fits and the separate saved-score
audit pass. Mean held-engine macro accuracy changes81.1866% ->81.3372%, while
Nyx recall falls90.8105% ->86.1872% and confusers assigned Nyx decrease74 ->53
among1,360 crops. This is still3.89706%, above the unchanged1% gate, and the
Nyx regression exceeds1pp. Improved fitted-bank accuracy99.2929% /zero fitted
confuser errors does not establish transfer or open-set wake acceptance.

The [complete evidence archive](assets-intent-source-balanced-sampling-v1.json)
contains all12 final checkpoints, complete intent scores/decodes, exact
probability vectors, crop/source metadata, sources/plan/runtime/commands,
preflight tests and original binary initializer. It retains the initial
wrong-directory failure before fitting and the corrected-path run separately.
The fresh anonymous download verifies all280 listed member hashes/lengths.
Full scalar features and the bank are separate byte-bound inputs. No audio,
credentials, full feature cache, firmware or device state is included.

To reproduce, use the pinned Python environment and upstream revision from the
plan. Start `launch_intent_sampling.py` with a fresh `--artifacts` directory and
`--repository`, `--upstream`, `--features`, `--bank`, `--initial-weights` paths.
`--bank` must point directly to the directory containing `bank.json` and
`bank.npz` (`compact-broad-bank-v1/bank` in the extracted workspace). The launcher
freezes committed sources and runs both training and read-only accounting.
`package_intent_sampling.py --help` documents evidence packaging and independent
public-download verification. The previous word/phone generalization archive
retains reference control records. No failed gate permits board deployment;
real-speaker, matched Hi ESP, P4 and long-call evidence remains outstanding.

## Matched-voice training acoustic neighbours

The [frozen diagnostic](../wakeword_p4/training/plans/phrase-neighbours-v1.json)
reads only the778 variant0 training paths /744 waveform hashes in the existing
broad bank. Original source WAVs establish fixed RMS activity proxies; cached
features supply centered cosine/DTW distances. Source audio is preserved at
its original rate. No alignment is played or passed to a classifier.

The [result](../wakeword_p4/results/phrase-neighbours-v1.json) retains129 voice
groups, collapses34 exact-waveform aliases and computes1,703 required pairs.
Only106 groups provide a distinct-positive reference; Piper has just one,
with Nick comparisons only. Of1,139 anchor/confuser comparisons,124 order a
confuser closer than the nearest distinct wake source. These are correlated
feature comparisons, not independent speakers, pronunciation errors or wake
false accepts. A fixed per-engine/class ranking yields38 review triplets,
including comparison cases without inversions. No human labels are available
for this queue, and no sample is filtered, relabeled or sent to new training.

The [independent accounting](../wakeword_p4/results/phrase-neighbours-v1-audit.json)
uses SciPy correlation costs and a full-table recurrence. All1,703 distances
agree within2.77556e-16; retained feature arrays exactly match the original
cache, and every comparison/cohort/review selection reconstructs. The full
inventory has zero exact waveform/intent conflicts. These checks validate
the measurement, without proving a perceptual collision or a cause.

The [complete archive](assets-phrase-neighbours-v1.json) includes source/plan,
launch/runtime, all selected cached feature arrays and distance matrices,
source identities, comparison/review metadata, tests and independent audit
code/logs. The anonymous download verifies all19 listed hashes/lengths. No
source PCM, full scalar cache, credentials, firmware or device state is included.

At source commit `8a490ac`, run `audit_phrase_neighbours.py --plan <frozen-plan>
--features <features-expanded-human335> --bank <compact-broad-bank-v1/bank>
--output <fresh-directory>`. Then run `verify_phrase_neighbours.py --artifacts
<that-directory> --features <same-features> --output <fresh-audit.json>`.
`package_phrase_neighbours.py --help` describes publication verification.
Original WAV/cache inputs remain byte-bound separately. Later listening or
human pronunciation review requires a separate step; no audio was played here.

## Frozen review queue: unprompted phoneme observations

The [fixed plan](../wakeword_p4/training/plans/review-phoneme-consistency-v1.json)
decodes all 97 original training WAVs needed by the 38 frozen triplets with
the pinned, unprompted `facebook/wav2vec2-lv-60-espeak-cv-ft` recognizer.
It uses whole 16 kHz PCM, CPU float32 and local weights only. The expected
text never enters inference; every frame logit and greedy CTC sequence is
retained. Model loading reports no missing, mismatched or unexpected keys.

In the [completed result](../wakeword_p4/results/review-phoneme-consistency-v1.json),
the 25 spectral-inversion triplets have 17 positive-closer phone predictions,
6 ties and 2 negative-closer predictions. The 13 noninversion triplets have
10 positive-closer predictions and 3 ties; there are no empty predictions.
Spectral distance therefore cannot automatically identify bad pronunciation.
The two negative-closer cases are Edge `en-IE-ConnorNeural/006.wav`
(intended “Hey Nyx.”; predicts spelled N Y X) and OpenAI `cedar/002.wav`
(intended “Hey Nicks?”; predicts the same phones as “Hey Next.”).
Kokoro `af_nova/000.wav` also shares its predicted sequence with a Next source.
These are fallible recognizer observations, not confirmed pronunciation errors,
actual waveform collisions, corpus-wide error estimates or replacement labels.
They do not establish an error in the GPU track's different training corpus.
No human listening result is recorded in this evidence, and no sample is
removed or relabeled. No new training or model promotion follows this study.

The [separate accounting audit](../wakeword_p4/results/review-phoneme-consistency-v1-audit.json)
checks original WAV hashes/rates/lengths and all 2,674,616 saved float32
logits, then reconstructs greedy CTC sequences, exact phone-distance orders,
cohorts and predicted-string collision groups. It uses a full-table edit recurrence,
`Fraction` comparisons and `itertools.groupby` CTC decoding. This verifies
saved-output accounting, not independent model inference or pronunciation.
All three focused tests passed before inference.

The [public evidence archive](assets-review-phoneme-consistency-v1.json)
contains all logits/predictions/comparisons, frozen source/plan/runtime/model
support JSON, raw logs/tests and the independent auditor. A fresh anonymous
download verifies all 23 listed member hashes and lengths. The 1.26 GB public
recognizer weights, original source audio and private listening material are
excluded; their required input identities remain pinned in the plan.

To reproduce inference at source `3acc9a0`, use the plan's pinned environment:

```sh
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  research/wakeword_p4/training/review_phoneme_consistency.py \
  --plan research/wakeword_p4/training/plans/review-phoneme-consistency-v1.json \
  --neighbours <phrase-neighbours-v1> --features <features-expanded-human335> \
  --model <pinned-phoneme-model> --output <fresh-study-directory>
python research/handoff/verify_review_phonemes.py \
  --artifacts <study-directory> --neighbours <phrase-neighbours-v1> \
  --features <features-expanded-human335> --output <fresh-audit.json>
```

`package_review_phonemes.py --help` documents packaging and public-download
verification. This study uses no playback, microphone, board, firmware or
reserved wake-quality scoring. Full Hi ESP parity remains unproved.

## Original GPU v30: silent full-app P4 probe completed

Coordinate Mac/P4 and RTX 4090 research in
[issue #13](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13).

### Cross-host report and floating-model diagnostic receipt

GPU commit `ee0becbf3d8716d14a977afbe52b2e5e5aa0ecee` reports the original v30
SavedModel on all 350 old Mac sources: 12/50 target triggers versus 11/50 INT8,
with the same three confusing negatives. The only float-only target is
`Moira-3-0.wav` (float maximum five-score mean .9440673232 versus INT8 .91328125
at the unchanged .91796875 threshold). The large recall gap is present in
these floating results too; this does not establish a cause or a best float
threshold.

The [Mac receipt](../wakeword_p4/results/gpu-v30-mac-gpu-diagnostic-receipt-v1.json)
checks report/script/model/training identities, all 350 source rows and 94940
original Mac output bytes. Every reported Linux INT8 event matches events
independently recomputed with integer five-score sums on the Mac trace. All
349 identical output streams have identical maximum five-score means. The
one differing stream, `Tessa-1-3.wav`, has maximum .0265625 on Mac and reported
.02421875 on Linux; both remain far below the fixed threshold.

The GPU floating script filters source events with `480*s-56160` although a
one-based score step `s` returns at `480*(s+1)-56160`. The receipt checks every
reported float and recomputed INT8 event: this offset error does not change
the source-event membership or any reported trigger count in these results.
It retains the corrected return offsets explicitly.

This is an independent accounting check, not independent float inference or
verification of Linux/float score arrays. Those full archives and original
floating weights remain requested through issue #13. Do not describe the
receipt as another execution of the floating model. Reproduce after fetching
the pinned GPU objects, using the original public full Mac study:

```sh
python research/wakeword_p4/training/audit_gpu_mac_diagnostics.py \
  --gpu-commit ee0becbf3d8716d14a977afbe52b2e5e5aa0ecee \
  --study "$FULL_MAC_STUDY" --output "$FRESH_RECEIPT_JSON"
```
The [result](../wakeword_p4/results/gpu-v30-p4-probe-v1.json) verifies the
original model, all 179 host/P4 output bytes and event step 144. All 12
preselected training-only reset traces reproduce: eight vary and four stay
zero. The rejected all-zero legacy Daniel fixture remains in the evidence.
These diagnostics are not recognition acceptance estimates.

Both raw-input arms complete 120-second idle measurements, with identical
SDK config and matching dependency versions/component hashes. Normalized
detector elapsed fractions are 8.4141%/8.7575%, with no new RX overflows in
the counted intervals. v30 minimum internal/DMA free memory is 69644 bytes
lower. Startup counts are already 765/749; v30 also runs its diagnostic.
CPU utilization, acoustic latency and active-call capacity remain unmeasured.
Both guarded tests restore the original app/service/Hi ESP listener; no
playback, microphone PCM recording or production-board operation occurs.

The [public archive manifest](assets-gpu-v30-p4-probe-v1.json) binds all 43
payload members and a fresh anonymous download. Extracted source regenerates
the fixture/report byte-for-byte and both capture audits. It includes the
model/reports, synthetic fixture, all diagnostic traces and numeric timing;
firmware, SDK config, board state/backups and raw UART stay private. See the
[runtime procedure](../panel_wake_runtime/README.md#completed-v30-silent-development-probe)
for reproduction commands, dependency pins and the 460800-baud USB fallback.
Full Hi ESP parity remains open.

## Frozen v30 on the complete Mac paired regression

The [2026-11-12 result](../wakeword_p4/results/gpu-v30-mac-regression-v1.json)
replays all 350 original public Mac sources at normal speed under the original
v30 policy and full-app 56160-sample zero prime. Every source WAV hash and PCM
CRC agrees with its retained Hi ESP P4 result. All 350 reset replays agree
exactly. The original host convolution and firmware sum/divide policies give
the same events on this set; neither threshold nor source selection changes.

| Reused synthetic metric | v30 host INT8 | Retained Hi ESP P4 |
| --- | ---: | ---: |
| Own target, any event | 11/50 | 48/50 |
| Own target, estimated end [-100,+800] ms | 11/50 | 32/50 |
| Same ordinary sentences, false-event clips | 0/100 | 0/100 |
| Same Nyx near words, false-event clips | 3/90 | 0/90 |
| ESP near words, pronunciation uncertain | 0/60 | 14/60 |

The v30 negative events are Tessa `Hey Next.` variant 3 and `Hey Nexus.`
variants 0/3. Target hits by voice are Daniel 0/10, Karen 4/10, Moira 4/10,
Samantha 0/10 and Tessa 3/10. Variants are correlated. Different target words
and phrase-specific confusers do not define one common false-wake population;
the 3/90 comparison uses identical Nyx-negative PCM. These are reused synthesis
intent labels, not verified pronunciations or new real-user recall. Hi ESP
comes from the prior standalone P4 run; new v30 inference runs on the host.
The separate 179-byte fixture proves only its measured P4 numeric agreement.
No acoustic, word-end latency or active-call acceptance follows here.

The [saved-output audit](../wakeword_p4/results/gpu-v30-mac-regression-v1-audit.json)
recomputes all source CRCs/frontend hashes and checks 94940 quantized output
bytes, integer firmware-policy events, host-policy events and critical counts
for every voice/variant cohort. It is accounting, not independent inference
or pronunciation truth. A fresh execution of extracted source/model/plan
reproduces the complete result JSON byte-for-byte and every output byte.
The [public evidence manifest](assets-gpu-v30-mac-regression-v1.json) verifies
all 24 payload members through anonymous download. Original source WAVs remain
in `paired-mac-regression-v1.tar.gz`; no audio is played or recorded.

The replay recipe was frozen at `00bd0d9`. Use the current research branch
for both commands below and the plan's pinned host versions:

```sh
python research/wakeword_p4/training/replay_gpu_mac_regression.py \
  --sources "$PUBLIC_MAC_PAIRED" --model-dir "$ORIGINAL_V30_PAYLOAD" \
  --plan research/wakeword_p4/training/plans/gpu-v30-mac-regression-v1.json \
  --output "$FRESH_STUDY"
python research/wakeword_p4/training/verify_gpu_mac_regression.py \
  --sources "$PUBLIC_MAC_PAIRED" --artifacts "$FRESH_STUDY" \
  --output "$FRESH_STUDY/audit.json"
```

The replay was committed before scoring; the auditor was added afterward.
`package_gpu_mac_regression.py --help` documents publication and anonymous
verification. All original PCM and full traces are retained; the source-relative
offset is the returning 480-sample detector block end after subtracting prime,
not the earlier frontend frame endpoint. Any prime events remain separate
(zero observed). Report this gap in [issue #13](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13)
for GPU development. No model promotion, threshold fitting or automatic source
relabeling follows from these results.

### v30 score separation: a threshold cannot repair this regression

The [frozen analysis](../wakeword_p4/training/plans/gpu-v30-mac-score-separation-v1.json)
at `c5cb028` computes necessary separation bounds directly from the complete
saved traces; it does not select or propose another threshold. The
[result](../wakeword_p4/results/gpu-v30-mac-score-separation-v1.json) checks all
350 output hashes and the original event booleans. Prime windows are all zero.
Two Tessa confusers (`Hey Next.`/`Hey Nexus.`, variant 3) each reach five
consecutive output bytes `[255,255,255,255,255]`: average 1275/1280, or
0.99609375, the maximum representable by this output contract. The highest
positive average is only 1274/1280. A threshold high enough to exclude those
confusers therefore admits **zero** positive clips. Both original host and
firmware averaging give this bound.

Additionally, six target-intent clips have only zero scores; at most 44/50
could ever trigger with a positive scalar threshold, even ignoring confusers.
The retained Hi ESP 48/50 remains a descriptive reused-data reference, not a
newly introduced acceptance gate. These are exact score-order facts about
the present INT8 traces. They neither establish source pronunciation nor
separate learned-model, quantization or platform causes.

The [separate audit](../wakeword_p4/results/gpu-v30-mac-score-separation-v1-audit.json)
uses an array-window view and independently constructed raw-score return
positions, and retains exact byte windows, source hashes and block ends for
the strongest positive/confuser witnesses. No new threshold grid or model
inference is performed. The pinned branches' `features()` and `Detector`
location-free ASTs are identical; native-library/runtime/platform equivalence
still needs actual cross-host comparison. That request, and a possible later
same-checkpoint float/INT8 diagnostic, are in issue #13. Existing v37 training
is not interrupted.

Reproduce using the preceding public full study (not the compact Git summary):

```sh
python research/wakeword_p4/training/analyze_mac_score_separation.py \
  --artifacts "$FULL_MAC_STUDY" \
  --plan research/wakeword_p4/training/plans/gpu-v30-mac-score-separation-v1.json \
  --output "$FRESH_SEPARATION"
python research/wakeword_p4/training/verify_mac_score_separation.py \
  --study "$FULL_MAC_STUDY" --analysis "$FRESH_SEPARATION" \
  --output "$FRESH_SEPARATION/audit.json"
```

For the AST comparison, fetch the pinned GPU branch objects first. Full inputs
are `study/result.json` and `study/scores.npz` in
`gpu-v30-mac-regression-v1.tar.gz`. No WAV, model execution, audio device,
board, firmware or reserved data is accessed by this analysis.

### Differentiable sequence scores for a future training experiment

`sequence_scorer.py` reuses the original window model's variables and frozen
BN, replacing its Flatten/Dense *application* with a sliding convolution of
the same Dense kernel. It computes an entire series in one graph while
retaining the original window model for eventual export. The inference graph,
parameter count, calibration and firmware are not changed. This is a research
helper, not an adopted training recipe or a new accepted model.

The [random-input probe](../wakeword_p4/results/sequence-scorer-random-five-gradient-v1.json)
uses the pinned teacher60 architecture, nonidentity frozen BN and two random
lengths (420/423 feature frames). All 22801 original parameters are shared.
Across 183 independent windows and 183 mature streaming outputs, maximum
probability delta is 8.94e-8. For the mean of five consecutive scores, all 15
trainable-variable gradients match five separate overlapping windows within
1.49e-8. Dynamic-length compiled `training=True` gives the same gradients as
eager execution and leaves every frozen BN variable unchanged. No optimizer
steps are taken.

The 150-frame teacher60 window uses 149 effective input frames because of
valid striding. For a complete feature stream `raw`, sequence scoring of
`raw[1:]` matches original streaming outputs from zero-based step49 onward;
scoring of `raw` instead matches window starts0,3,6,... . This mapping retains
the known one-frame phase; the helper does not invent future context or alter
microphone speed. In the future, a trainer could directly differentiate a
five-score objective rather than assuming one terminal positive score creates
enough temporal support. Loss/label/interval design and actual gains remain
unproven.

```sh
python research/wakeword_p4/training/probe_sequence_scorer.py \
  --upstream "$PINNED_MICROWAKEWORD" --output "$FRESH_PROBE_JSON"
```

The upstream revision must be `4665173cd35f1cff9a61e06fc427f124766c488e`.
Upstream `to_streaming_inference` changes source layers' `mode` in place,
so the probe exports a separate model with the same weights. Do the same if
using the shared sequence graph during training; converting its original
window model in place invalidates the shared graph. Before adoption, check
the helper against actual trained weights and existing complete-source streams,
including any remaining native numerical drift. Random graph/gradient parity
does not prove recognition, an isolated failure cause, export parity or P4
performance. This probe reads no WAV, checkpoint, reserved set or device.

### Mac/Linux frontend witness

GPU branch `40e77bf` executed the original 350 PCM sources on Linux CPU.
Its report has 101 different frontend hashes and one different output stream
(`rendered/Tessa-1-3.wav`, source index 283). The GPU issue update reports two
different output bytes and unchanged fixed-policy events. This is close
agreement, not byte identity. It does not explain away the retained 11/50
versus 48/50 regression, or establish the cause of the frontend difference.

The [Mac witness](../wakeword_p4/results/gpu-v30-mac-frontend-witness-v1.json)
exports that exact source under the original Mac versions. Both its frontend
hash and every output byte reproduce the original archived Mac run. It also
independently invokes the interpreter with saved input tensors and checks
that all 305 output bytes agree. The
[public asset](assets-gpu-v30-mac-frontend-witness-v1.json) contains:

- `witness/arrays.npz`: complete little-endian UINT16 frontend `[916,40]`,
  INT8 quantized inputs `[916,40]`, the actual 305 invoked input tensors
  `[305,1,3,40]`, and UINT8 outputs `[305]`. The final frontend frame is not
  part of an invoked triplet, as in the unchanged streaming detector.
- `witness/report.json`: input/model/reference/array hashes, original PCM CRC,
  prime/tail lengths, actual package/Python/platform versions, frontend wheel
  metadata and native binary hash. No native binary or original audio is
  included.
- `source/`: exporter plus the unmodified frontend, detector and policy source.

The archive was anonymously downloaded and all six payload members verified.
It contains no firmware, credentials, device state, microphone recordings or
WAVs. Reproduce with the original Mac package versions and public inputs:

```sh
python research/wakeword_p4/training/export_mac_frontend_witness.py \
  --sources "$PUBLIC_MAC_PAIRED" --model-dir "$ORIGINAL_V30_PAYLOAD" \
  --reference "$FULL_MAC_STUDY" --output "$FRESH_WITNESS"
```

`FULL_MAC_STUDY` is the `study` directory in the public full regression archive.
The exporter verifies its fixed result, score, plan, manifest and baseline
hashes; it does not relax or spoof the pinned Mac version checks.

For Linux diagnosis, compare its fresh frontend with `frontend_uint16` and
report differing-element count, first coordinates/values and maximum absolute
delta. Compare the *actually invoked* INT8 inputs as well. Finally, reset the
same original interpreter and feed each `invoked_input_int8` tensor directly,
checking all 305 bytes against `output_uint8`. This separates frontend/input
differences from interpreter differences without re-exporting the model or
changing thresholds. Exchange results in
[issue #13](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13).
