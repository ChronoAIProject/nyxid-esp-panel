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
