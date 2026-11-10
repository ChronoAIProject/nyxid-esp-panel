# GPU research cross-reference

Latest observed GPU commit: `c0c4a50`; see the dated snapshots below.
The Mac input-control result is `../wakeword_p4/results/hiesp-input-acoustic-v1.json`.

Observed 2026-11-11 at GPU branch commit
`d338e7a`:
[research-hye-nyx-voice-gpu](https://github.com/ChronoAIProject/nyxid-esp-panel/tree/research-hye-nyx-voice-gpu).
Read its subsequent commits at the beginning of each new experiment cycle.
This is an evidence snapshot, not a claim that the branch remains unchanged.

The GPU branch imports earlier Mac research from `b59c05e` and now records:

| Work | Completed evidence | Remaining evidence |
| --- | --- | --- |
| RTX 4090 / PyTorch setup | CUDA training and export on a random-array classifier | This model does not measure wake-word recognition |
| Piper CUDA synthesis | 13,920 waveform clips with full file/manifest audit | Broader voice coverage and P4 acceptance |
| Kokoro synthesis | 3,220 waveform clips / same 28 voices, multi-speed hash/voice audit | More correlated views do not establish new speakers |
| MicroFrontend features | v2: 68,560 correlated sequences / 17,140 source WAVs | Further training experiments and P4 comparison |
| TensorFlow GPU training | Initial through negative-window-mining-v9 runs complete; v9 INT8 validation rejected | None accepted |
| Released P4 QEMU | Pinned Linux emulator runs; minimal IDF5.5.4 firmware builds | Application/timer markers absent after25s; boot and WakeNet inference unverified |
| CUDA Qwen synthesis | 360 waveform-QA-accepted clips; 298 training-only ASR-agreement selections | Description conditions are not independent speaker identities; pronunciation uncertainty remains |
| Real room augmentation | 270 MIT RIR WAVs used in v4/v5; base data preservation and renderer regression verified | v4 validation rejected; no generalization improvement established |
| Broader negative training | MUSAN music 8.547 h / noise 6.226 h; v5 total 18.7485 h continuous training exposure | Validation exposure remains 1.20481 h; no final acceptance |
| Official pretrained-feature desktop reference | Corrected batched mel/embedding consistency, complete CUDA cache, and four rejected trained heads | No MCU export or accepted detector |
| Reserved negative exposure | 10.745 source hours / 73 speakers, archive/source audit | No model scores read; independent positive holdout still needed |
| New Mandarin development coverage | AISHELL speaker-disjoint extraction audit; frozen v6 has1 event/6.914 validation hours | Upper95% .686/hour; far-field/TV, positive recall and matched Hi ESP evidence missing |
| Unscored paired challenge | Corrected 468 WAVs / 26 unused international Kokoro presets; pairing/isolation/ASR screens | Pronunciation uncertainty, acoustic and matched Hi ESP scores |

The `6cf7a35` commit adds an audited feature pipeline, pinned Linux GPU
environment and an `official64` MicroWakeWord training command for 40,000 steps.
Commits `d42d0b4` through `9c90954` publish actual detector results and a validation
phase diagnosis. This is a separate
10 ms MicroFrontend track; the current Mac continuation uses an 80 ms speech
embedding and two classifier heads. Their model inputs and trained weights are
not interchangeable. Compare actual exported detector metrics before choosing
a P4 adapter; keep each frontend/feature/calibration identity together.

The published reports contain hashes and commands; these new source WAVs and
trained wake weights are not attached to that branch as downloadable assets at
this snapshot. The current Mac continuation can already be reproduced
from the separate [research prerelease](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/tag/research-hey-nyx-20261110)
and this branch's `assets.json`, including completed features, checkpoints and
portable plans. Its GPU environment smoke still needs execution on the target.

The two Piper runs use the same source voice ID sets: training 672, validation
112, testing 120. Whole sets were compared directly against the existing Mac
metadata. GPU resampling uses SciPy polyphase instead of the earlier
torchaudio path, so identical corpus intent does not imply identical waveform
bytes. Re-generated renderings of the same voices do not provide fresh speaker
acceptance. Preserve source voice partitions when importing any new waveforms.

The GPU branch's existing test split has already been used in earlier research
and is correctly described there as regression. The current untouched final
English/Mandarin negative corpora remain separate; no new detector scores on
those final corpora were read for this cross-reference.

The first GPU INT8 model (`06a05930…`, 63,032 bytes) was rejected. No validation
grid point met both required constraints; its fallback threshold is 0.992. On the
existing regression it reaches 87.55% Piper recall but only 27.08% Kokoro
(57.32% engine macro); continuous negatives have 1 event / 1.3582 source hours.
These correlated synthetic variants do not establish real-user performance.
The eight-clip float-stream/INT8-stream diagnostic reproduces resets but does
not establish MCU numeric equivalence.

The subsequent 48-variant Kokoro validation diagnosis isolates a **window-phase
mismatch**, rather than attributing all recall loss to quantization. A valid
kernel-5/stride-3 convolution consumes 179 of 180 input frames: shifting the
non-streaming window by one 10 ms frame reduces maximum float difference from
0.94655 to 6.56e-7. The GPU branch derives this phase before validation and
has rerun the matched recipe. Do not transfer this architecture-specific fix
to the different 80 ms embedding frontend without separate evidence. No
matched Hi ESP result is published yet.

The aligned v1b candidate (`c866e74c…`) is also rejected: regression engine
macro recall 61.64% (Piper 79.53%, Kokoro 43.75%), confusing false detections
1.384%, zero continuous events / 1.3582 source hours. Its 95% false-event upper
bound is still 2.2057/hour. The GPU rerun is not claimed bit-exact and selects
different weights/thresholds; the metric change does not isolate alignment's
causal effect. The v2 run combines expanded speed views, equal engine sampling
and latched-event validation after causal warmup. Treat it as a multiple-change
experiment. Our local channel-EQ study preserves source waveform speed.

Balanced v2 (`2a163828…`) is now also rejected: 57.604% regression engine macro
recall (Piper 65.625%, Kokoro 49.583%), 0.7386% confusing false detections and
zero continuous events / 1.3582 hours, still a 2.2057/hour upper bound. It
records 90 premature positive events. Expanded speed views also change the
regression population, so scores are not a controlled comparison to v1/v1b.

The challenge generator's tokenizer originally pronounced `ESP` as a word.
The GPU branch preserves/withdraws that version and creates an unscored
`Hi E S P.` replacement with manifest `40b4ed9e…`. Whisper tiny.en agrees on
32/78 Hey Nyx, 28/78 Hi ESP and 236/312 negative clip texts; it is an ASR
consistency screen, not phonetic truth or a wake-recognition test. No clips
were removed/relabelled. The 10.745-hour reserve remains unscored as well.

Before repeating Qwen preparation, note that this local snapshot already
contains 960 training-only Qwen clips, their feature views and completed
full/screened continuations. Its screened two-head combination still has
46/50 paired wakes and 6 English / 2 Mandarin false events and was rejected;
see `research/wakeword_p4/results/embedding-temporal-qwen-v3.json` and the
original assets. New GPU VoiceDesign conditions may still be useful, but
re-generating old conditions does not add independent speaker acceptance.

Local work complements that run with generic five-band channel augmentation
of the existing 80 ms embedding training features. It preserves original
sample count/rate/speed and does not fit a room filter from microphone audio.
This is a separate hypothesis; importing unqualified GPU weights or adding
more correlated TTS variants is not an acceptance result.

Commit `6a3b08b` adds a primary-source review in
`research/wakeword_p4/BEST_PRACTICES.md`. The new GPU Qwen source has 72 positive
and 288 negative clips across 24 description conditions. An independently
declared training-only ASR policy retains 30 positives and 268 negatives;
all parent files, labels and exclusions remain auditable. ASR agreement is
not pronunciation ground truth and can bias accent coverage. The v3 extension
adds 1,192 correlated feature views while an explicit byte comparison preserves
the full v2 binary prefix and record list, including validation/regression.
No v3 training or recognition gain is reported.

The GPU work now prioritizes realistic acoustic and negative coverage before
another source-only run. Its pinned official-notebook MIT RIR source is audited,
but not yet used for training. Official precomputed negative features advertise
a 20 ms step, versus this GPU track's 10 ms step, so they have not been imported.
Our local 80 ms embedding model is another distinct input contract. Reuse raw
audio and verify the appropriate frontend rather than interchanging arrays.
The corrected paired challenge and 10.745-hour negative reserve remain unscored.
Local work therefore focuses on matched physical speaker-to-panel measurement;
it does not duplicate the GPU RIR preparation or read its reserved scores.

Subsequent commits `901577e` and `ff4282c` complete the measured-room and
expanded-negative experiments. V4 adds 28,600 training renderings with measured
RIRs and -5..10 dB noise: a combined intervention. Its INT8 validation macro
recall is 49.974%, below v2's 57.109%. V5 retains v4's generated features and adds
1,062 MUSAN training ambient recordings, raising source exposure to 18.7485 h.
At the validation-only .95 threshold it reaches 59.010% source macro recall
(Piper 75.521%, Kokoro 42.500%). This is not a controlled statistical gain;
Kokoro remains below v2's 46.667%. Both new models have zero continuous events
in 1.20481 validation hours, still a 2.4865/hour one-sided 95% upper bound.
Neither reads regression/final scores or qualifies for acceptance. Their
eight-clip reset/numeric diagnostics do not establish general equivalence.

The next GPU priority is positive-source quality/generalization, with a possible
pretrained-feature diagnostic. The existing Mac snapshot already supplies six
TTS engines and a frozen pretrained-embedding detector. Rather than regenerating
those voices, `waveform_inventory.py` now verifies **11,604 original WAVs**
(10,070 training, 1,534 validation) from the downloaded assets. It preserves
source-PCM keyword endpoints, text and engine-qualified voice tokens, verifies
WAV hashes/headers and source/voice isolation, and omits every derived feature
frame coordinate. See the handoff for commands and the sanitized verification
report. It excludes testing, private captures and the separate 960 extra Qwen
sources. These reused development voices are not a fresh evaluation population.
Extract the destination frontend from the raw WAVs; never import embedding
arrays or old `keyword_frame` values into the 10 ms track. New synthesis isn't
required to test this already available source diversity.

GPU Qwen v6 now completes both validation and development regression. Its
validation source-macro recall is 69.948%, but development regression is only
61.120% (Piper 73.906%, Kokoro 48.333%), with one continuous false event in
1.35816 source hours. The 95% upper bound is 3.4929/hour. It remains rejected;
the validation gain does not establish generalization, and the final challenge
and 10.745-hour negative reserve remain unscored.

The separate desktop openWakeWord diagnostic uses the pinned official float
backbone, not the local quantized streaming embedding. Its CUDA13 ONNX loader
stays separate from TensorFlow's CUDA12 environment. The initial batched cache
was rejected because TF32 drift and a batch-global mel ReduceMax changed
features. TF32 is now disabled, the derived mel graph reduces only non-batch
axes, and generated waveform/sequential consistency checks pass. The completed
cache contains 94,418 sequences / 8,230,203 embedding frames, including 88,719
training and 5,699 validation sequences. This establishes cache consistency,
not wake recognition or MCU compatibility.

The first desktop head, upstream-architecture-only v2 and training-only
eight-window hard-negative-mining v3 all remain rejected. Validation macro
recall is respectively 30.026%, 42.370% and 51.380%; all have zero continuous
events in only 1.20481 validation hours, whose upper bound remains 2.48647/h.
Phrase attribution identifies Hey Nick / Next / Mix among confusing negatives.
The positive-phrase audit does not isolate all misses to command-followed
speech or one spelling. A v4 learning-rate control uses the pinned upstream
1e-4 target with warmup/hold/cosine; all 40,000 schedule values agree within
1.3553e-20. The completed v4 control selects step40k/threshold.992 and reaches
50.781% validation macro recall (Piper28.229%, Kokoro73.333%), below v3's
51.380%. Zero events/1.20481h still gives upper95%2.48647/h; it remains
rejected and regression/final fixtures are unscored. Negative weight
scheduling and low-loss filtering remain separate recipe deviations.

The latest audit finds a concrete ambient-sampling coverage imbalance in both
GPU trainers: uniform row sampling gives human speech only 2.12% of draws
despite 17.20% of source duration, while short noise rows receive 82.23% of
draws despite 33.21% of duration. The optional duration-weighted sampler passes
a 100,000-draw coverage check (17.163% human, expected 17.201%). Its forthcoming
MCU-architecture v7 control has launched and changes only this sampling choice
against v6. No
recognition gain or universal preference for duration weighting is established.
These GPU checks do not justify changing the Mac detector or mixing 10 ms
MicroFrontend arrays, desktop float arrays and local 80 ms INT8 embeddings.

For matched development comparison, this branch now exports all350 unchanged
paired Mac synthetic WAVs plus their already measured standalone-P4 Hi ESP
numeric results as `paired-mac-regression-v1.tar.gz`. It includes50 Hey Nyx,
50 Hi ESP and250 ordinary/near-word negatives across the same five voices and
two correlated normal-speed variants. Every WAV hash, decoded PCM CRC,
512sample inference coverage and source/result identity is checked. The two
unrelated previous acoustic controls are excluded. This enables GPU models
to use the exact existing development comparison without needing WakeNet on
Linux; it is not the independent final challenge or a new acceptance set.
See the [portable comparison instructions](README.md#portable-paired-pcm-baseline).

Subsequent GPU v7 quantized validation is also rejected: source-macro recall
67.865% (Piper74.896%, Kokoro60.833%), below v6's69.948%, at threshold.97.
It has zero continuous events/1.20481h, whose upper95% bound remains2.48647/h.
No development regression or final scores are read. The eight-clip numerical
comparison has maximum float/INT8 drift.25409; it is not general parity or
MCU numerical evidence. Duration weighting corrected a sampling imbalance
without establishing an improved detector.

A stronger ASR check materially changes the training pronunciation hypothesis.
On the same200 training Piper WAVs, tiny.en's alias-aware positive agreement
is32/100, while pinned small.en CUDA FP16 reaches80/100 (36/40 standalone,
44/60 commands). Positive transcripts starting Hey Next decrease45to7.
This contradicts using tiny.en disagreement alone to claim widespread bad
Piper pronunciation or discard those WAVs. The two models share the Whisper
family and neither proves phonetic ground truth. Piper's intended Nyx/Nix/
Nicks phonemes are identical /nɪks/; merely respelling them would not change
that generator's intended input. No audio, labels or evaluation rows change.

The stronger check also agrees with62/72 Qwen training positives and270/288
negatives. A union with the earlier tiny-selected subset preserves every old
source and adds34 positives/10 negatives, yielding64+/278-. The v8-small MCU
run tests only that training-source addition against v6, preserving row-based
ambient sampling and all other settings. Its264 new renderings preserve the
complete old binary/index prefix and unchanged human/ambient/evaluation rows.
The full audit has99,212 sequences from17,482 source WAVs. Actual INT8
validation is pending; independent final fixtures remain unscored. The Mac
branch should not duplicate these Qwen/ASR experiments or assume v8 is eligible
for flashing before its results and model artifact are available.

The v7 validation attribution reproduces all1584 positive hits and20
confusing-negative triggered variants without new inference or thresholds.
Piper recall is81.458/84.167% in quiet/microphone-floor variants versus
66.250/67.708% in colored/real noise plus synthetic-reverb variants. Kokoro
stays58.333–63.333%, including61.667% quiet; all12 premature events are Kokoro.
This separates observed source and acoustic-condition deficits without proving
their causes. The local80ms embedding continuation now freezes a different
matched synthetic-versus-measured response study: same original/noise draws,
matched response energy and additional-view counts, no microphone fitting.
Its48-pair training-only pilot passes; full extraction and subsequent fitting
remain separate from GPU v8 and cannot be treated as recognition acceptance.

Continue exchanging results by source commit, model/feature hashes, frozen
validation policy and downloadable artifacts. Keep the two active research
branches independent; compare or cherry-pick explicit changes after reviewing
their source. Production firmware and the dining-room panel remain excluded.

```sh
git fetch origin research-hye-nyx-voice-gpu
git log --oneline -8 origin/research-hye-nyx-voice-gpu
git show origin/research-hye-nyx-voice-gpu:research/wakeword_p4/GPU_RESEARCH.md
```

GPU commits `fa61787` through `0987834` complete and reject v8-small: INT8
validation source-macro recall63.594% at.9 (Piper64.271%, Kokoro62.917%),
6 premature events, confusing-negative false rate.5682%, and zero ambient
events/1.20481h (upper95%2.48647/h). No regression/final scores were read.
The broader Qwen source union did not produce an accepted detector.

The new pinned AISHELL audit retains9,722 read-Mandarin utterances from28
speaker-disjoint sources:2,756 train,5,598 validation,1,368 development
regression.13 defective/untranscribed WAVs remain quarantined. At its old.8
threshold, frozen v6 has **one** event/6.913985 validation source hours,
observed.144634/h and upper95%.686126/h. Original utterance gaps/alignment
padding are excluded from exposure, and state-reset prefix checks pass for
all16 scored speakers. This is not zero events; the latest prose correction
agrees with the authoritative full JSON. No threshold/retraining or independent
final scoring occurred. Read speech cannot establish room/TV performance.

Locally, all10,070 paired80ms measured-room feature views now pass their full
audit. The four dense/temporal fitting plans are frozen in
`embedding/plans/measured-room-continuation-v1.json` before fitting. This
remains a distinct frontend/experiment from the GPU10ms models. No microphone
recordings, evaluation sources, audio speed changes or new acceptance claim
enter this training stage.

That local 80 ms study has now completed. Its measured treatment is rejected:
44/50 existing paired positives versus matched-control 46/50 and Hi ESP 48/50;
both continuations retain one ordinary-sentence event and two English events.
No inferior model was flashed and no new acoustic playback occurred. Preserve
this as a failed matched experiment, rather than repeat its RIR substitution
or infer a general claim that real-room augmentation cannot help. Its exact
features/checkpoints and portable fitting plans are published as the two
`measured-room-*-v1.tar.gz` research assets. Independent final fixtures remain
unscored; the overall Hey Nyx objective remains unmet.

GPU commits `8c00e47` through `d338e7a` now reject the training-only negative
window-mining control v9: validation engine macro63.646% (Piper83.125%,
Kokoro44.167%) at.5, with33 premature events. The pooled78.796% figure is
dominated by Piper variants and does not establish a source-balanced gain.
Zero ambient events/1.20481h still gives upper95%2.48647/h. No regression or
final scores were computed. Do not repeat this as an untried mining recipe.

Espressif's pinned `esp-develop-9.2.2-20261109` Linux release lists the P4
machine and a minimal IDF5.5.4 firmware compiles, but the25-second run prints
only `Adding SPI flash device`. No application or timer marker appears.
This is a pending boot diagnosis, not an available board-free WakeNet baseline;
optimized DSP instructions and numerical/recognition parity remain unverified.

The new local training-only diagnostic scores384 frozen audio hashes, with
four canonical original/environment views each. It preserves68 exact-audio
duplicates across the full source-file inventory as same-role spellings,
without inventing opposing labels or extra independent recordings. The
diagnostic finds26 false clips/768 negative training variants, and a separate
source-RMS check finds delayed file-end markers in some standalone TTS clips.
A metadata-only continuation now tests correction of68 extra-view markers
across34 audio hashes. PCM, features, base-view markers and PTQ sampling stay
unchanged. This is distinct from new acoustic augmentation or loss weighting.
The continuation now completes without a deployment improvement. Its frozen
equal mean selects .4, validation engine-macro94.4851% versus retained matched
control94.6845%, each with one event/7.30378h. Existing development positives
are44/50 versus historical original mean46/50 and Hi ESP48/50. Near-word
false clips20/90 versus original23/90 are a tradeoff; ordinary1/100,
English1/5.549h and Mandarin0/1.294h remain unchanged. Candidate rejected,
no flash/playback or final scoring. This partial augmented-view metadata
experiment does not establish whether full phonetic alignment would help.

### GPU snapshot through `168e236`

The newer GPU branch completes and rejects v10 SpecAugment and v11 Mandarin
addition. v10 improves source-macro validation to77.6823% at .9, but frozen
development recall drops to57.8906% with60 premature Kokoro events. Its
zero AISHELL events/6.913985h gives upper95%.433286/h for that read-speech
corpus only. v11 reaches only40.0781% validation macro at .994 and has no
qualified operating point. Neither reaches final acceptance; do not duplicate
these recipes as untried experiments.

The board-free P4 boot diagnosis has advanced: ROM/application/FreeRTOS/timer
markers now pass twice, with an independently checked ~98ms interval. This
supersedes the earlier absence-of-serial-output observation. Exact ESP-SR2.6.1
and the Hi ESP model enumerate, but model creation faults on an unsupported
PIE instruction (`esp.ld.128.usar.ip`) in the official QEMU build. No detect
call completes, so simulated Hi ESP recognition remains unavailable. The
PSRAM workaround is probe-only and must not enter physical-board firmware.

### GPU snapshot through `ed2b6b1`

The GPU branch now adds speaker/session-disjoint DiPCo far-field conversation
sources. At frozen thresholds, S05 validation yields v6 zero and v10 one event
in.754488h; respective upper95% rates3.97055/h and6.28753/h are too weak for
acceptance. Read-Mandarin bounds must not be generalized to this conversation.
S09/S10 development and official evaluation recordings remain unscored there.
v13 adds only training S02/S04 with the original recipe; results remain pending.

The frequency-only SpecAugment control v12 is rejected at65.7552% validation
source-macro, threshold.98. This does not establish time masking as the cause
of v10's premature events. The branch also pins official Hey Jarvis v2 for
MCU pretrained-transfer research, but its30/60-channel architecture differs
from the current32/64 model. No transfer or numerical parity is yet claimed;
the Mac branch should not independently duplicate that reconstruction.

Locally, a separate original-data negative BCE weight20 control also fails:
94.5595% versus retained weight5 control94.6845% mean validation macro, two
versus one continuous event/7.30378h. All prefit sampled windows and PTQ
representatives match; no endpoint correction or mining was combined. The
candidate is rejected before development/final scoring or physical flashing.
Its full checkpoints remain available as `negative20-continuation-v1.tar.gz`.


### GPU snapshot through `72fc264`

The matched MCU scratch/official-pretrained experiment now reports v15 transfer
validation macro recall75.7292% at the frozen.95 threshold, with three premature
positive events. Its reused development regression reaches67.6042% macro
(Piper78.5417%, Kokoro56.6667%), with68 premature Kokoro events. Zero events
in1.35816 ambient source hours still gives upper95%2.20573/hour. These scores
do not meet Hi ESP equivalence. At that same frozen threshold, AISHELL produces
two events/6.913985h (upper95%.910588/hour), while DiPCo S05 produces zero in
only.754488h (upper95%3.97055/hour). Neither establishes an acceptable general
household false-wake rate. Preserve the failed Mandarin gate.

The next GPU recipe audits sparse human sampling and adds training-only DiPCo
S02/S04 streams (1.26550 original hours), retaining all validation/regression
rows. Its paired stratified training remains pending at this snapshot; do not
repeat it locally or treat feature preparation as recognition improvement.
The Mac continuation instead isolates physical Hi ESP raw-versus-AEC/AGC input
under matched dependencies/optimization, with unchanged PCM playback speed and
call capture path. No new model/threshold or reserved-final scores are needed
for that diagnostic.


### GPU snapshot through `454eb32`

The paired human-sampling controls finish: row-weighted v16 reaches 59.4531%
validation engine macro, while category-stratified v17 reaches 70.4167% at
the same .95 threshold. V17's existing development regression is 66.1458%
macro, with 40 premature events and one ambient event / 1.358162 h (upper
95% 3.49286/h). At the frozen threshold it has zero AISHELL events / 6.913985 h
(upper 95% .433286/h), but one DiPCo S05 event / .754488 h (upper 95%
6.28753/h). The read-Mandarin result does not establish household acceptance,
and the candidate remains rejected. The branch preserves the conversation
false-event context without reusing that validation audio for training.

V18/v19 now test 8 or 16 training-positive prefix negatives per batch, retaining
the original evaluation window, source partitions and remaining RNG sequence.
Both runs are pending at this snapshot. This directly overlaps the possible
prefix-supervision follow-up; the Mac side will not duplicate it. Its current
physical Hi ESP input comparison continues without changing any detector
threshold, replay speed, provider setting or training data.


### GPU snapshot through `78c8264`

The prefix experiment completes: 8 prefix negatives per batch (v18) is rejected
at 66.5104% validation macro; 16 (v19) reaches 78.3073% at the frozen .95
threshold. Existing development regression improves to 69.4271% macro, with
17 premature events versus v17's 40 and v15's 68. Zero ambient events /
1.358162 h still gives upper 95% 2.20573/h. This remains a research candidate,
not an accepted detector or a claim that prefix supervision always improves
recognition. Both experiments preserve the original evaluation window.

Frozen v19 has zero AISHELL events / 6.913985 validation hours (upper 95%
.433286/h) and zero DiPCo S05 events / .754488 h (upper 95% 3.97055/h).
Newly scored, previously assigned development regression streams yield zero
AISHELL events / 1.723096 h and zero DiPCo S09/S10 events / .712196 h; their
bounds remain 1.73858/h and 4.20633/h. These cannot be relabelled untouched
final holdouts after scoring. Independent final audio remains unscored.

The GPU branch is downloading the officially hashed LibriSpeech dev-other
for preassigned development coverage and audits speaker separation before
feature extraction. No dev-other model result is available at this snapshot.
No v19 model/calibration was identified among the currently listed research
prerelease assets. Physical v19 acceptance cannot be inferred from its JSON
scores or attempted by substituting another frontend's weights.


### GPU snapshot through `c0c4a50`

The officially hashed LibriSpeech dev-other development extraction completes:
33 speakers, 5.121185 original hours, no speaker-ID intersection with existing
human/final manifests. At the previously frozen .95 threshold, v19 has one
event (upper 95% .926322/h), and v17 has two (upper 95% 1.22936/h). Both fail
the recorded .5/h bound on this read-speech corpus. The v19 event overlaps
ordinary text containing “on the nets”; this is a hypothesis source, not
phonetic causal proof or permission to move validation audio into training.

No .tflite/checkpoint is tracked in the GPU research tree at this snapshot.
The [model handoff contract](GPU_MODEL_HANDOFF.md) records the actual
`teacher60`/context/validation metadata boundary and required model/report/
synthetic INT8 probe identities before attempting physical integration.
The current Mac raw-input engine bracket proceeds independently.


### GPU snapshot through `ce601c4`

The training-only near-word expansion (v20) preserves existing evaluation
bytes but lowers validation macro recall to 68.8802%, versus v19's 78.3073%,
at the same .95 threshold. Its worst-source near-word false fraction is 1%,
versus v19's .875%. It is rejected before development/final rescoring. This
rejects that particular data/sampling recipe, not near-word training generally.

Expanded validation calibration finds no acceptable v19 threshold in its
recorded grid: the dev-other false event remains through .994 as positive
recall falls. Its five-score mean reaches the representable uint8 ceiling
255/256, so merely raising the threshold cannot rescue this model at that
event. A separately fixed 15-score smoothing experiment also fails validation
at 55.3906% macro with median latency 280 ms; it is rejected without development
or final rescoring. The Mac physical plan keeps its already frozen model and
event policy; these findings are not permission to tune against acoustic scores.

The GPU track now starts v21, a widened `teacher120` architecture with 70,801
parameters versus 22,801. Its Net2Wider initialization agrees with the original
float teacher on 16 numerical probes within 5.37e-7; the classifier is then
reset for training. This is neither quantized MCU byte equivalence nor Hey Nyx
quality evidence. Training is pending at this snapshot. Do not duplicate the
GPU capacity experiment locally or assume the future `teacher120` artifact
fits the existing `teacher60` metadata/runtime boundary. Independent final
audio remains unscored, and the overall goal remains active.
