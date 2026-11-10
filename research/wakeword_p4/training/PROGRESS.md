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
- Frozen fullphrase96 regressions now use the exact completed Hi ESP PCM
  baselines. At its unchanged validation-selected 0.992 threshold, English
  continuous false wakes fall from old long335's 13 to one / 5.549 h; Hi ESP
  has zero. Mandarin has zero / 1.294 h for both. However, paired Mac any-event
  recall is only 22/50 versus Hi ESP's 48/50. Shared ordinary sentences are
  0/100 for each, and Hey Nyx's own near-word subset triggers 2/90. These
  correlated synthetic regressions show a remaining voice-generalization gap,
  not acceptance. Newly reserved testing remains unread. See
  `../results/fullphrase96-matched-regressions-v3.json`.
- A further phoneme pilot inspected 12 training positives and 12 negatives per
  existing engine, 144 clips total. It retains raw phones and source hashes,
  including uncertain vowel/consonant predictions. Piper command suffixes use
  the original duration-derived keyword metadata; other sources here are
  standalone phrases. The small, differently worded groups do not establish an
  engine quality ranking or justify automatic relabeling/filtering. See
  `../results/cross-engine-phonemes-v3.json`. A future data-quality ablation
  must be recorded separately and leave the running recipe unchanged.

- Saved checkpoints can now be exported explicitly with `--export-only`,
  `--steps 0` and matching `--initial-weights`. This performs no optimization
  or checkpoint selection. All 20 model-weight arrays are bit-exact for the
  old fullphrase96 20k and current style 10k/20k/30k exports. Fresh seed-726
  training-only representative sampling can change INT8 quantization; every
  resulting model hash therefore receives a new validation calibration.
  An aligned ten-clip float window/stream audit of the old 20k export differs
  by at most 4.21e-6, while INT8/float differences reach 0.329. Export fidelity
  alone is not recognition acceptance. Invalid export-only CLI combinations
  are rejected before creating output or reading weights.
- Quantized calibration now supports an explicit threshold grid and a
  `continuous` diagnostic policy: maximize mean-engine validation recall with
  observed ambient FAR <=0.5/h. The default historical grid and strict policy
  remain. `viable_operating_point` continues to mean that the original strict
  near-word constraints pass, even when the continuous policy is used. The
  full old-model/default-policy regression preserves selected metrics, counts,
  per-engine recalls and identities exactly; four grid macro means differ by
  1.11e-16 from unordered floating-point summation. Invalid/nonfinite/duplicate
  threshold inputs are rejected before audio/output access.
- Re-exporting the old fullphrase96 20k checkpoint with its own newly frozen
  0.9 continuous-policy threshold improves mean-engine validation recall to
  87.95%, but paired Mac recall is only 32/50 versus Hi ESP's 48/50. English
  and Mandarin negatives each have one event, versus zero for the reference.
  It is rejected. Different checkpoint and fresh INT8 quantization prevent
  interpreting this as a threshold-only causal comparison. See
  `../results/fullphrase96-20k-continuous-v3.json`.
- The style/multilingual 30k training schedule has completed. Explicit 10k,
  20k and 30k exports use a prespecified 19-threshold validation sweep, including
  thresholds below 0.5: weighted BCE outputs are not calibrated probabilities.
  The first two exports reach 85.04%/88.39% mean-engine validation recall at
  0.15/0.1, with 2/3 events in 7.324 h. Their longest above-threshold ambient
  runs are 0.72/0.93 s, ruling out an always-high/latch artifact on these scores.
  Near-word subgroup errors remain about 10.42% maximum, and the strict gate
  fails. The 30k sweep completes at 84.53% mean-engine recall, so the
  validation-only policy fixes the 20k export and 0.1 threshold. Its paired
  regression improves to 44/50 versus Hi ESP's 48/50, but shared ordinary
  sentences trigger 2/100 versus zero. Identical long English and Mandarin
  PCM produce five and two events respectively, versus zero for Hi ESP.
  Reject this replacement; recall/FAR remain worse. See
  `../results/style-multilingual-checkpoints-v3.json`. Newly reserved testing
  and full-app/acoustic acceptance remain unmeasured.

- A controlled PTQ experiment kept the style20k weights bit-exact and increased
  representative training sequences from 100 to 1,000. Mean-engine validation
  recall decreases from 88.39% to 86.51%, with three ambient events in both
  cases. A 60-source-clip aligned float/stream audit still agrees within
  5.07e-6, while selected maximum float/INT8 deltas grow from 0.437 to 0.658.
  More representative samples do not repair this candidate. These differences
  alone do not establish their causal contribution to overall recognition.
  The inferior export does not proceed to testing. See
  `../results/representative-corpus-ablation-v3.json`. The historical default
  remains 100; nonpositive/noninteger CLI values fail before data/output access.
- The original checksum-pinned training archive contains another 155 eligible
  speakers, after excluding every existing partition and continuous regression
  speaker. Their 63.085 h (62.114 h actual speech) are now training-only.
  Every new PCM/feature checksum passes, no indexed FLAC hashes overlap, and
  the chained feature view preserves all 195 previous human rows plus original
  synthetic/environmental targets and partitions. A temporary real-FLAC
  fixture exercises both default overlap rejection and explicit exclusion,
  including group IDs without speaker identities and alias screening. See
  `../results/expanded-human-training-data-v3.json`.
- Two prespecified same-initial-weight fine-tunes are running from private
  source snapshots: current data versus the 155-speaker extension, each with
  feature masking enabled, 20k steps, learning rate 1e-4, negative weight 5,
  existing 50%/pool4 training-only mining, uniform generators, 335 frames and
  100 representative calibration sequences. Explicit 10k/20k exports will
  receive the same 19-threshold quantized validation sweep. This pair isolates
  additional human data within the new augmentation recipe; it does not
  isolate the effect of masking versus earlier training. No newly reserved
  testing scores or production-board changes occurred. Development-board
  status still verifies the original app, service and Hi ESP listener.

- Both prespecified 20k fine-tunes and all four 10k/20k INT8 exports/calibrations
  completed. The expanded-human 10k export is frozen by validation only:
  threshold 0.7, 88.67% mean-engine recall, three events / 7.324 h. Its 20k
  checkpoint lowers recall to 84.08%; additional steps do not improve this
  selection. The current-data control's 10k/20k recalls are 88.09%/86.57% at
  their own 0.4 thresholds. New reserved testing remains unread.
- The frozen expanded10k candidate improves the old style20k long-negative
  regression from five English/two Mandarin events to one English event /
  5.549 h and zero Mandarin events / 1.294 h. Hi ESP has zero on both exact
  PCM baselines. Paired Mac recall is 42/50 versus Hi ESP's 48/50, with one
  ordinary-sentence false wake / 100 versus zero. Reject it as a replacement:
  this is a recall/FAR tradeoff, not achievement of Hi ESP performance. See
  `../results/specaug-controlled-finetunes-v3.json`. An initial paired launcher
  used the wrong manifest and was rejected before scoring; the corrected run
  uses the completed 352-clip baseline's exact manifest including two controls.
- A separate full-application research adapter now builds with the pinned
  release dependencies, matching model/training/calibration hashes and the
  existing single I2S/AEC/AGC producer and ordered FIFO. The old style20k model
  was used for integration diagnostics only, not candidate acceptance. Its
  6,608,784-byte app boots, selects the original service and reports Hey Nyx
  listening; a synthetic Daniel wake-plus-request fails to trigger. Its exact
  source WAV triggers host INT8 inference (maximum smoothed score 0.95625 at
  frozen threshold 0.1), so microphone/domain/runtime diagnostics are needed.
  A new Daniel Hi ESP control also fails, while the previously verified Hi ESP
  source passes the real wake/request/reply/quiet/cleanup test. Do not attribute
  the failure solely to the custom model or claim matched acoustic recall.
  The original app, service and Hi ESP listener were restored and verified
  after the unsuccessful experimental command. Diagnostic UART progress and
  resource measurements are being added in the separate runtime checkout.

- The current full-app diagnostic records nonzero microphone peaks up to 17,742
  while every five-second maximum smoothed score remains zero. Its inference
  progresses at the expected real-time rate, mean about 2.6 ms / 30 ms step,
  with at least 11,748 bytes of wake-task stack free. These are diagnostic
  observations, not final CPU/latency acceptance. A bounded exact-PCM startup
  self-test now compares 132 host output bytes, spanning 0..247, through the
  complete firmware's own frontend and model. It is model-bound and optional,
  and its detector is discarded before microphone listening. Hardware result
  is pending. Eight runtime artifact/self-test boundary tests pass.
- A new prespecified official64/335-frame run is active on the completed
  expanded-human corpus: 30k steps, 10k checkpoints, learning rate 3e-4,
  negative weight 5, uniform engines, 50%/pool4 mining, feature masking and
  100 representative sequences. This changes topology/initialization from the
  transfer60 warm-start lineage and is not a causal width-only comparison.
  Keep the same quantized validation policy and freeze the choice before
  existing regressions. New reserved test scores remain unread.

- The complete-firmware numeric startup self-test passes: 132/132 output bytes
  match, including scores 0..247. The test detector is destroyed before live
  listening. A temporary 30-second copy of the existing detector input then
  captured exactly 480,000 microphone-path samples, 3,000 packets from sample
  zero with no missing samples. Replaying that PCM on the host with the exact
  85,920-sample rounded prime also produces all-zero scores. Offline gain
  factors 1/2/4/8/16 do not repair it. This case is not established runtime
  numerical divergence; the received acoustic/input distribution is failing.
  Tiny.en hears `Hey mix` in both source and capture, so pronunciation remains
  uncertain and the failed new Daniel Hi ESP control must not be ignored.
  No speed, board gain, AEC or buffering change was made. All four full-app
  experiments restored and verified the original app/service/Hi ESP listener.
  Temporary audio-copy code was removed from working source. See
  `../results/full-app-runtime-diagnostic-v3.json`.
- The official64 run has saved its 10k checkpoint and continues the fixed 30k
  schedule. A separate private checkpoint workflow exports and calibrates each
  prespecified 10k/20k/30k artifact as it becomes available, then freezes the
  quantized validation choice before the existing paired/English/Mandarin
  regressions. It never scores newly reserved tests or declares acceptance.

Keep each run in a fresh private artifact directory. Record source hashes and
failed results. Choose candidates from validation only; reserve new unseen
voices/conditions and real recordings for subsequent acceptance.

- The training-only pronunciation audit now covers all 6,851 prespecified
  positive/Hey-negative source clips. It is still running. A fresh feature-view
  helper implements the frozen screening ablation, with no label changes and
  exact validation/testing record and chained-human-target preservation. Six
  boundary tests pass, including incomplete audits, label tampering, held-out
  selections and empty positive engines. No screened training has begun.
- Six matched natural-speed synthetic acoustic fixtures have been generated
  with three OpenAI voices for both Hi ESP and Hey Nyx plus the same immediate
  request. Marin is an existing training voice; echo/sage are existing validation
  voices, so this is a development diagnostic, not independent holdout recall.
  Physical Hi ESP wake/request/reply checks are running on the restored original
  development app before trying a newly selected candidate. No production board
  or reserved test voice is involved.
- Physical Hi ESP controls complete: Marin passes wake, immediate request,
  complete reply and eight quiet seconds; Echo and Sage do not wake. Tiny.en
  hears the intended source phrases in all six files, which does not establish
  phonetic acceptance or remove the failed controls. The existing rejected
  style20k app is now being measured with the matched Marin fixture for a
  diagnostic full-app comparison. See `../results/matched-acoustic-controls-v3.json`.
- A fixed-weight calibration-boundary experiment follows upstream's two-pixel
  min/max convention instead of artificial constant frames. All 22 stored H5
  arrays and the complete exported TFLite bytes match the historical style20k
  control exactly. Redundant validation/fidelity/regression scoring was stopped:
  this convention does not explain or fix that artifact's recognition failure.
  The historical default is unchanged. `audit_export.py` now identifies an
  explicitly supplied INT8 artifact separately from its float weights/report.
  See `../results/representative-input-bounds-v3.json`.
- Before completing the phoneme audit, a matched follow-up plan was recorded:
  unfiltered versus screened training data, both initialized from expanded10k,
  identical 10k-step recipes and prespecified 5k/10k checkpoints. It launches
  only after the complete identity/preservation checks and nonempty positive
  engine check. Quantized validation selects across four exports before the
  existing development regressions. No fresh testing or acceptance is implied.
- Official64/335 training and all three INT8 calibrations/regressions completed.
  Validation freezes 10k: mean-engine recall87.50% at0.5 versus84.53%/85.97%
  for20k/30k. Paired recall41/50 is below Hi ESP48/50, with seven English
  events/5.549h and one Mandarin event/1.294h versus zero reference events.
  Reject the replacement. See `../results/official64-expanded-v3.json`.
- All6,851 pronunciation audits completed. The frozen view excludes1,883
  positive and three ambiguous-negative sources, preserving every held-out row
  and original feature target. It retains very few Mac/Kokoro positives, so the
  screen may discard legitimate accents. Both matched10k fine-tunes are active;
  four5k/10k INT8 exports are selected only by unchanged validation. See
  `../results/training-pronunciation-screen-v3.json`.
- A development-only30s stage capture locates the matched Marin failure:
  source0.96875 and raw mic0.99609375 each trigger; communication-AEC clean
  and subsequent AGC both score zero. Both transports start at sample0 and
  finish at480,000 with no loss; playback reference peak is zero. Across six
  leading phase shifts0..400samples, raw mic always triggers and AEC output
  never does (max0.0046875). This demonstrates loss at the preprocessing stage
  for this phrase, not defective communication AEC or population-level recall.
  Private capture code is removed, clean build passes and original app restores.
- The opt-in runtime now feeds idle custom detection from raw mic samples of
  the existing single I2S producer, using its own MicroFrontend NR/PCAN. The
  processed preroll/call FIFO and communication AEC/AGC, Opus and normal speed
  are unchanged; default Hi ESP keeps its original input. Two real acoustic
  wake/request/reply/hangup/rearm rounds are running with the diagnostic old
  style20k artifact. Its rejected offline metrics still prevent acceptance.
  See `../results/aec-wake-input-diagnostic-v3.json`.
- Both raw-input full-app Marin rounds pass: each records the complete immediate
  request, exactly one user turn and one reply, eight quiet seconds without
  another turn, own-call cleanup and successful rearming for the next wake.
  The runner restores/verifies the original app. This fixes the demonstrated
  input-path integration failure; the old style20k model remains rejected for
  its offline recall/FAR. No human/production acceptance is claimed.
