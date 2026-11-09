# Active goal: Hey Nyx comparable to Hi ESP

Started 2026-11-10 at the user's explicit request. **Active; not achieved.**
The production dining-room board is excluded from research changes.

## Evidence required

Compare the shipped `wn9_hiesp` / `DET_MODE_95` against a frozen Hey Nyx export.
Use matched generators/voices, room/noise conditions and speech levels. Each
engine receives its own phrase; shared ordinary-speech/environment negatives
are identical. Report phrase-specific confusing negatives separately, rather
than equating two different sets of near words.

Measure recall by voice/condition, continuous false events/hour and confidence
bounds, and detection latency relative to the actual keyword end. Offline RMS
label estimates do not establish precise phonetic latency. Paired synthetic
tests are development regressions, not proof of real-user performance.

Production acceptance requires real acoustic tests on the development panel,
including the user's voice, distance/noise variations, immediate post-wake
requests, playback echo, repeated rearm, reconnect and long calls. Compare Hi
ESP and Hey Nyx using the same microphone/AEC/AGC path. Report failures and
uncertainty; do not equate a low synthetic false rate with Hi ESP quality.
Long continuous negatives must provide enough exposure to assess the desired
<=0.5/hour range; zero events in about six measured hours only bounds that
particular audio distribution, not all real environments.

Keep one 16 kHz I2S -> AEC -> AGC producer, the existing ordered wake FIFO,
24 kbps Opus and 20 ms packets. Do not change live speech speed, add another
microphone reader, duplicate buffered speech, mute all playback, or suppress
transcripts to hide echo. Measure full-app flash, CPU, RAM and AEC stability.

## Completed diagnostics

- The v2 four-engine candidate is rejected; see [RESULTS.md](RESULTS.md).
- `audit_export.py` compares 36 selected validation clips after aligning the
  first convolution's streaming phase. Float windows and float streaming agree
  to a maximum absolute score difference of about `2.12e-6`. Individual float
  streaming/INT8 differences reach about 0.195; that is a score diagnostic,
  not proof that quantization caused the recognition failure.
- New paired Mac fixtures contain 350 quiet/noisy renderings. The fixed five
  source voices have already informed development. Each target has 25 source
  positives; render variants are correlated. No claim of a fresh holdout or
  real-human acceptance is made.
- The first bare `Hi ESP` TTS fixtures were unsuitable: a diagnostic recognizer
  repeatedly heard abbreviation expansions such as “especially.” Regenerated
  positives explicitly spell `E S P`. ASR is a diagnostic, not phonetic ground
  truth. The six ESP-specific near-word texts in this fixture version are still
  reported separately with unvalidated pronunciation.
- A 14-clip transfer smoke passed every PCM CRC. The benchmark app was then
  replaced with the verified original development app; its ELF identity, local
  service connection and Hi ESP listening were checked. This verifies the
  measurement/recovery plumbing, not recognition quality.
- The corrected full Hi ESP baseline completed all 352 CRC-verified transfers
  and verified original-app restoration. Keyword-only paired recall by any
  event is 48/50 variants (25 correlated source phrases), shared negatives
  0/190, and other wake phrase 0/50. Both previously used request controls
  trigger. The unvalidated ESP-specific near words trigger 14/60 and are kept
  separate. Narrow estimated-end-window hits are only 32/50 because the energy
  endpoint is not a phonetic boundary. Do not claim precise latency or replace
  any-event recall with this endpoint artifact. Standalone mean inference is
  2.126 ms per 512 samples, maximum 2.947 ms; this is not full-app CPU load.
- Wide128 and max-pooling exports also finished, with quantized validation
  recall 44.75% and 65.10% at their separately selected operating thresholds.
  Both fail the recall target. Sanitized details are in
  `../results/training-v3-contexts.json`; paired baseline aggregates are in
  `../results/hiesp-paired-baseline-v3.json`.
- Checking any-score versus narrow estimated-end scoring on those validation
  details does not explain their poor Hey Nyx recall: context203 adds only 5
  hits, wide203 only 2, max203 none, among 2,040 positive variants. None is an
  early-only hit. Preserve both scoring definitions in paired comparisons.
- The sixth TTS source completed all 1,288 clips, and six-engine features now
  contain 45,108 sequences. A 24-clip training/validation pronunciation pilot
  retained 6/12 positives and 11/12 negatives under tiny.en's imperfect exact
  text rule. It is a diagnostic only: raw generator labels are preserved,
  no ambiguous positives are turned into negatives, and test splits were not
  read. This is not phonetic certification of the generated source.
- A pretrained ESPHome v2 encoder import is numerically verified on 32 selected
  validation negatives and nine original-phrase controls. Original Hey Jarvis
  classifier only: float window/stream delta <=1.14e-6 on controls, maximum
  dequantized-float/original-INT8 delta about 0.043. The original three positive
  control scores are not Hey Nyx results. A separate one-step training/export
  smoke passed. Source license/provenance and detailed results are recorded in
  `../results/transfer-import-v3.json` and the README.
- The 203-frame, official64 context run finished. Quantized validation has no
  operating point satisfying the per-source confusing-negative constraint:
  fallback threshold 0.98, 50.49% overall recall, 61.17% mean-source recall,
  3.125% maximum-source confusing false rate, and 0 ambient events in 1.234 h.
  No test or Mac regression audio was read by this calibration. Reject it as a
  production candidate; longer context alone has not solved the problem.
- A new Mac corpus contains 672 source clips from 16 voice families / 24
  presets. Families were split before synthesis (12 training, 2 validation,
  2 testing); old regression voices are excluded. Accent variants of one
  family never cross splits. These IDs do not establish independent human
  speakers. Diagnostic ASR read only training/validation, and does not relabel
  their words or establish pronunciation correctness.

## Further diagnostics and experiments

- The original short Hi ESP baseline above is complete, including verified
  original-app restoration. The separate long continuous run is still active.
- Four- and five-engine equal-sampling training and quantized validation-only
  calibration completed. Both remain rejected: 43.53% and 17.57% overall recall,
  with no operating point satisfying the synthetic per-source precision target.
  They use identical context203 settings
  and seeds, in separate artifact directories. Five-engine features contain
  39,956 sequences. Do not attribute improvement to adding Mac voices if the
  sampling change has not been controlled separately. No final comparison set
  has been evaluated for either model.
- Six-engine `transfer60` and random-initialization control completed with
  identical 149-frame context, 20k steps, learning rate 1e-4, negative weight 5,
  equal engine sampling and no feature masking. Both reset the classifier.
  Quantized validation recall is 41.14% and 15.43% at their selected thresholds;
  neither is usable. Transfer helps this recipe but does not achieve acceptance.
  Source pretraining may have seen voices related to our synthetic data; do
  not claim end-to-end speaker-disjoint pretraining or human acceptance.
- A training-only RMS span diagnostic found 71.6% of the OpenAI positive clips
  exceed 1.22 s, with estimated active spans up to 2.89 s. A 149-frame window
  plus up to 270 ms of positive crop offset may omit part of these phrases.
  This is a plausible limitation, not a demonstrated sole cause. Spans for all
  engines are in `../results/training-spans-v3.json`.
- Long-context features are complete: 45,108 sequences / 31,700,300 frames,
  using 4 s lead (3.8–6.5 s on jittered variants) so a 335-frame model can warm
  before each phrase. This changes frontend history as well as model context;
  do not attribute any improvement to context alone. PCM speed is unchanged.
  The original 2 s default renderer still reproduces the checked quiet/noisy
  paired fixtures sample-for-sample.
- The 149-frame warm-start finished and remains rejected. Its quantized
  validation fallback has 77.05% recall, but one ambient false event in 1.234 h
  and 4.17% maximum-source confusing false rate. No viable operating point
  exists on the measured grid. See `../results/training-v3-warm.json`.
- `model-transfer6-long335` finished and fails the original quantized validation
  constraints: fallback recall 46.66%, maximum-source confusing false rate
  3.125%, zero ambient events in 1.234 h. See `../results/training-v3-long335.json`.
  New runs preserve every measured checkpoint rather than discarding
  unselected weights.
- A **separate, explicitly diagnostic** threshold policy for that exact model
  maximizes validation recall subject only to observed ambient <=0.5/hour.
  Threshold 0.7 was frozen before paired/long-negative evaluation; it does not
  replace the original failed calibration. It reaches 48/50 paired synthetic
  positives, but triggers on 15/190 shared negatives, including `Hey Mix`,
  `Hey Next`, `Hey Nexus`, and `The next thing is ready`. On exact-PCM
  test-clean negatives it has 13 events / 5.549 h (2.343/hour). Reject it.
  See `../results/long335-ambient-diagnostic-v3.json`. This new negative source
  has now informed development; future measurements on it are regressions.
- Controlled 149-frame warm-start runs compare ordinary sampling with 50%
  hard-negative selection from four-times larger training-only candidate
  pools. Initial weights, context, step budget, learning rate and source
  balance match. Both are rejected after quantized validation. The mined run
  reduces observed ambient events from 2 to 1 at separately selected thresholds
  but lowers recall; it does not prove an overall improvement. See
  `../results/training-v3-mining149.json`.
- Auditing continuous evaluation found that the old preparation padded every
  utterance to a 10 ms hop while the benchmark WAV did not include that padding.
  Exact features have now been rebuilt from complete, hash-verified WAVs in a
  fresh corpus. The Hi ESP WAVs and benchmark manifest are unchanged. A complete
  495.86 s WAV matches whole-array extraction bit-for-bit; the legacy features
  had 17 extra frames. This is a comparison-plumbing correction, not an
  explanation of the earlier classifier failures. Legacy validation/model
  artifacts are preserved. See `../results/continuous-feature-audit-v3.json`.
- The existing synthetic negative vocabulary lacks isolated wake components.
  Separate Kokoro (336 clips) and Edge (552 clips) corpora now test `Hey` alone, `Nicks/Nix/Nyx` alone
  and ordinary uses without the full wake phrase. Full Hey Nyx/Nix/Nicks aliases
  remain positive. Generator voice-family partitions and normal speed are
  unchanged. No test scores are used to create this vocabulary. Fresh long-lead
  features are complete: 48,660 sequences / 34,366,706 frames; these are two more
  corpora from existing engines, not two new engines.
- Before training on them, the old long335 model at its frozen 0.7 diagnostic
  threshold triggered on 130/528 validation partial-word variants. Standalone
  `Nicks/Nix/Nyx` accounts for most of these errors: this model demonstrably
  does not require the full wake phrase on those synthetic examples. See
  `../results/partial-vocabulary-before-training-v3.json`. It does not prove
  that every long-audio false event has the same cause.
- Downloading the official LibriSpeech train-clean-100 source for more varied,
  training-only human negatives. Preparation verifies its published MD5,
  rejects overlap with all existing speaker partitions, screens wake aliases
  and records actual WAV/features/provenance. The 6,387,309,499-byte archive now
  passes official MD5 `2a93770f6d5c6c964bc36631d331a522`, SHA-256
  `d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2`.
  Preparation reserves 64 training, 16 validation and 16 testing speaker IDs
  before feature extraction; all must be disjoint from existing dev-clean and
  continuous test-clean speakers. Newly reserved testing scores remain unread.
- The new human preparation completed: 26.546 h training, 6.090 h validation
  and 6.455 h testing (explicit separators included), with 64/16/16 speaker IDs.
  No transcript aliases or overlaps were found. A fresh feature view preserves
  all original synthetic features and these human split assignments. The next
  335-frame full-phrase warm-start recipe was fixed before scoring the new
  holdouts. It combines isolated-component negatives, more human negatives and
  optional training-only mining; any improvement cannot be attributed to only
  one of these changes without another controlled ablation.
- Two 30,000-step, 335-frame runs are now active on this feature view: one
  warm-starts the old 15,000-step float checkpoint, the other uses random
  initialization of the same `transfer60` topology. Data, learning rate 3e-4,
  negative weight 5 and 50% training-only mining match; host thread limits are
  6 and 3 respectively. This is not a timing comparison. Select candidates on
  validation only, and inspect original confusing words and new component
  negatives separately so additional easy negatives cannot hide old errors.
- The longer-context P4 numeric probe now builds with per-model input
  quantization and 16 KiB variable-state storage. A 1,054-step vector set includes
  validation speech and spans the candidate's complete 0..255 output range.
  Board execution is pending: the Hi ESP long-baseline runner still owns USB,
  and must restore the original app before any subsequent probe.
- LibriSpeech test-clean (CC BY 4.0) passed the official MD5 checksum and has
  recorded SHA-256 `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`.
  Transcript alias screening found no exclusions. It provides 5.403 h of public
  speech plus 0.146 h of explicitly counted separators, across 40 speakers.
  Exact PCM and continuous features are ready. Hi ESP is now running this
  longer negative baseline on the development board; original-app restoration
  is required when it finishes. Never open a second owner of its USB port.

- Quantized calibration can now explicitly enforce each engine/corpus negative
  group with `evaluate_stream.py --separate-negative-corpora`. This keeps old
  confusing words and newly added component negatives separate even when both
  use the same TTS engine. Historical default selection is unchanged. A small
  deterministic harness verified that 100 easy clips cannot hide a false wake
  in the original corpus. Current fullphrase96 quantized candidates will use
  this option, fixed before their export/calibration; no new test speakers have
  been scored. Calibration also reports progress through clips and long audio.
- `../run_probe.py` now makes the numeric app-only flash/capture/restore
  procedure reproducible without committed device identities. Mocked success
  and firmware-error paths both restore and verify the original app. Actual
  execution is still pending the existing long Hi ESP baseline's restoration;
  no second USB owner was opened.
- The warm-start run's 10,000-step float checkpoint is not viable: at its
  selected fallback 0.992 threshold it has 69.87% mean-engine recall, 7.29%
  maximum-engine confusing-clip false rate, and one event in 7.324 validation
  hours. A targeted validation diagnostic finds 4/528 partial-word variants
  above the fixed diagnostic 0.7 threshold, but OpenAI `Hey Mix` and `Hey Next`
  remain saturated confusions. These are float results; the older partial-word
  report used INT8 inference and is not a controlled comparison. Unprompted
  tiny.en transcribes those validation source phrases as expected; it does not
  certify phonetic correctness. Full training and its scratch control continue.
  See `../results/fullphrase10k-diagnostic-v3.json`.
- The same-topology scratch control's 10,000-step float checkpoint also fails:
  fallback 0.97 has 56.62% mean-engine recall, 5.21% maximum-engine confusing
  false clips and one event in 7.324 validation hours. Both runs continue their
  fixed schedules; do not infer final quantized performance from these points.
- A separate Mandarin evaluation corpus is now prepared from the pinned,
  CC BY 4.0 Google FLEURS source: 409 validation utterances / 1.294 h and 945
  testing utterances / 3.126 h, explicit separators included. Official
  partitions are retained. The export has no individual speaker IDs, so its
  8/19 continuous groups must not be reported as independent speakers. Original
  LFS hashes, duplicate/alias checks, all PCM/feature hashes, sample accounting
  and frontend-prefix equality passed. No detector scores were read; testing
  remains reserved. See `../results/mandarin-corpus-v3.json` and
  `fleurs_negatives.py`. This data is not added to the active training runs.
- Reviewing the untested UART speed option found that 6 Mbaud can outrun the
  measured ~2.13 ms/512-sample inference without flow control. It was removed
  before hardware use; 3 Mbaud remains experimental and rebuilt successfully.
  An eight-fixture short/long regression preserves exact historical PCM and
  expected detection offsets. Hardware execution still waits for the original
  long baseline's restoration; its 1.5 Mbaud run is unchanged.
- The frozen old INT8 long335 model also fails on Mandarin validation: one
  event / 1.294 h (0.773/h, upper 95% 3.667/h). Its English/Chinese long-audio
  events all occur after model warmup, so an initial decision delay does not
  explain them. Testing remains unread. See
  `../results/mandarin-long335-diagnostic-v3.json`.
- Preparation now optionally supports the official, separately checksum-pinned
  Mandarin training shard, rejecting duplicates against both held-out groups.
  Its download is in progress; the active runs are unchanged. Chaining human
  feature extensions exposed a reference bug: reusing `additional-human` could
  redirect the earlier corpus's rows into the newer one. Prior rows are now
  anchored through their original feature view. A two-layer byte-level
  regression verifies both corpora, preserves explicit group versus speaker
  identities and rejects duplicates. No running training source was edited.
- Hi ESP's full exact-PCM English baseline completed: 40 speakers / 5.549 h,
  zero events, every transfer CRC verified, and original app/local service/
  wake listening restored. The rejected old long335 model has 13 events on
  the identical PCM. The zero-event reference's one-sided 95% upper bound is
  0.540/h, not proof of zero production false wakes. See
  `../results/continuous-hiesp-vs-long335-v3.json`.
- Long335 P4 numeric/resource checks passed on 1,054 steps: exact output
  matches, 31,788 tensor bytes used, 64+16 KiB arenas allocated, median 708 us,
  and 13,756 stack bytes free. One raw frontend value differs by one unit;
  actual quantized inputs/outputs match. Original app restored. See
  `../results/p4-long335-probe-v3.json`. This is not recognition acceptance.
- 3 Mbaud now passes an eight-fixture short/long regression, including 495.86 s
  continuous PCM: CRC, sample/call counts, events and exact offsets agree with
  historical 1.5 Mbaud results. Original app restoration passed. Fully spelled
  Hi ESP near words then completed and restored: 31/60 clips triggered,
  predominantly `Hi E S B`, `Hi E S T` and `Hi A S P`. ASR remains uncertain
  about some consonants; preserve this limit. The earlier self-imposed 1%
  near-word gate is stricter than the measured reference. Report it separately
  from matched ordinary-speech false wakes and actual acoustic acceptance;
  do not declare equivalence from different near-word vocabularies alone.
- Fullphrase96 warm training/export finished. Its validation-selected INT8
  fallback 0.992 has 74.99% mean-engine recall, 7.29% maximum-corpus confusing
  false rate, 0/528 component negatives and one event / 7.324 h. Reject it;
  newly reserved testing speakers remain unread. The event's source transcript
  contains no full wake phrase. See `../results/fullphrase96-validation-v3.json`.
- The Mandarin training shard completed: 3,246 utterances / 9.906 h including
  separators, 9.726 h actual speech, and no duplicate source audio against
  reserved validation/testing. A chained feature view verifies every original
  file reference and adds only training groups. A new 945-utterance OpenAI
  corpus balances positive/negative prosody across existing training voices;
  synthesis and all WAV hashes passed. Original sources plus these additions
  are rebuilding with unchanged 4 s leads. Verify old feature bytes before
  launching the separately recorded next recipe. A combined-data improvement
  cannot be attributed to one addition without an ablation.
- Hi ESP Mandarin validation completed all eight groups / 1.294 h with zero
  events. Every PCM identity matches the rejected old long335 diagnostic,
  which has one event. The reference's short-exposure upper 95% bound is
  2.315/h, so this does not establish low production risk. The original app,
  Nyx connection and Hi ESP listener were restored; the runner exited normally
  and released USB. See `../results/mandarin-hiesp-vs-long335-v3.json`.
- The matching fullphrase96 random-initialization control completed 30k steps
  and INT8 validation. Its fallback 0.97 has 56.84% mean-engine recall versus
  the warm run's 74.99% at its own fallback 0.992. The scratch model has one
  event / 7.324 h and 5.21% maximum-corpus confusing false rate; both models
  fail the strict near-phrase gate. Neither is accepted or installed. New
  testing scores remain unread. See
  `../results/fullphrase96-scratch-validation-v3.json`.
- The combined-data feature rebuild finished. All 48,660 original synthetic
  sequences compare byte-for-byte by source/variant; original labels, metadata
  and partitions agree. All 136 old human feature targets and metadata are
  preserved, and 77 environmental feature files are bit-exact. The 3,780 new
  TTS variants and 59 Mandarin groups are training-only. Source hashes and
  the full identity check are retained privately.
- The balanced OpenAI 72-clip tiny.en pilot strictly recognized only 6/36
  positives, with 28 Next substitutions. A separate, pinned multilingual
  wav2vec2 phoneme CTC diagnostic instead yields canonical `h eɪ n ɪ k s` on
  11/36 and `h eː n ɪ k s` on 5/36, alongside other vowel variants and a few
  m/final-t confusions. This argues against treating the lexical transcripts
  as direct phonetic labels, while leaving real pronunciation uncertain.
  Generator labels remain unchanged. See
  `../results/style-phoneme-diagnostic-v3.json` and `audit_phonemes.py`.
- The pre-recorded style/multilingual335 recipe is now training from a private
  immutable source snapshot: transfer60, 335 frames, warm initialization from
  fullphrase96, 30k steps, six CPU threads, unchanged sampling/weight/mining
  settings. Only the balanced training TTS and disjoint Mandarin training data
  are added. Quantized validation, frozen-threshold regressions and full-app
  acoustic/resource/AEC tests remain outstanding. This run is not acceptance.

Keep each run in a fresh private artifact directory. Record source hashes and
failed results. Choose candidates from validation only; reserve new unseen
voices/conditions and real recordings for subsequent acceptance.
