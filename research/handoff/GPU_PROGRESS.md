# GPU research cross-reference

Observed 2026-11-10 at GPU branch commit
`ff4282c`:
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
| TensorFlow GPU training | Initial, aligned, balanced-v2, measured-room-v4 and expanded-negative-v5 runs complete | None accepted; v3 Qwen feature experiment remains untrained |
| CUDA Qwen synthesis | 360 waveform-QA-accepted clips; 298 training-only ASR-agreement selections | Description conditions are not independent speaker identities; pronunciation uncertainty remains |
| Real room augmentation | 270 MIT RIR WAVs used in v4/v5; base data preservation and renderer regression verified | v4 validation rejected; no generalization improvement established |
| Broader negative training | MUSAN music 8.547 h / noise 6.226 h; v5 total 18.7485 h continuous training exposure | Validation exposure remains 1.20481 h; no final acceptance |
| Reserved negative exposure | 10.745 source hours / 73 speakers, archive/source audit | No model scores read; independent positive holdout still needed |
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

Continue exchanging results by source commit, model/feature hashes, frozen
validation policy and downloadable artifacts. Keep the two active research
branches independent; compare or cherry-pick explicit changes after reviewing
their source. Production firmware and the dining-room panel remain excluded.

```sh
git fetch origin research-hye-nyx-voice-gpu
git log --oneline -8 origin/research-hye-nyx-voice-gpu
git show origin/research-hye-nyx-voice-gpu:research/wakeword_p4/GPU_RESEARCH.md
```
