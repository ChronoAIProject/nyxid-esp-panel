# GPU research cross-reference

Observed 2026-11-10 at GPU branch commit
`9c90954`:
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
| TensorFlow GPU training | Initial, aligned and balanced-v2 40,000-step runs complete | All rejected; Qwen source preparation pending synthesis |
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

Continue exchanging results by source commit, model/feature hashes, frozen
validation policy and downloadable artifacts. Keep the two active research
branches independent; compare or cherry-pick explicit changes after reviewing
their source. Production firmware and the dining-room panel remain excluded.

```sh
git fetch origin research-hye-nyx-voice-gpu
git log --oneline -8 origin/research-hye-nyx-voice-gpu
git show origin/research-hye-nyx-voice-gpu:research/wakeword_p4/GPU_RESEARCH.md
```
