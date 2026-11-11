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
ESP and Hey Nyx using the same physical microphone and conditions, documenting
each detector's preprocessing. The opt-in custom detector now reads raw idle
mic samples; both retain the same AEC/AGC processed call path. Report failures and
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

- Matched pronunciation-control/screen training and all four INT8 validations
  finished. Validation freezes screen5k, but its paired recall is43/50 versus
  reference48/50. English has1event/5.549h and Mandarin2/1.294h versus0/0.
  Reject this replacement. All reserved testing remains unread.
- The prespecified fixed-weight3/5/7-step smoothing ablation selects7steps
  at0.15 by validation. Mean-engine recall90.64% versus89.93% for5steps is
  a modest gain; paired recall43/50 and long false counts1/2 are unchanged.
  Metadata now binds smoothing in both regression consumers and opt-in firmware;
  old calibrations retain5steps, invalid explicit fields fail. Public calibration
  reproduces the private snapshot (only two floating means differ <=2.3e-16).
  See `../results/quality-screen-smoothing-v3.json`.
- New quality/smooth7 complete-app numeric startup check matches132/132 bytes.
  Two physical Marin wake/immediate-request/reply/quiet/hangup/rearm rounds pass,
  with one user and one assistant turn each. Original development app, service
  and Hi ESP listener are restored/verified. This is integration evidence only:
  the classifier still fails recognition acceptance. No production board touched.
  See `../results/full-app-quality-smooth7-v3.json`.
- A distinct pretrained speech-embedding route is being assessed before another
  tiny-model fine-tune. Exact float reconstruction agrees within5.34e-5 on28
  training-only/control sequences. INT8 export is390,560bytes from1,330,312;
  mean embedding cosine.9743/min.938 is a feasibility diagnostic, not wake recall.
  A bounded standalone P4 memory/timing probe is building. No classifier, native
  frontend, whole-app fit or release provenance acceptance is established.

- Cached pretrained embedding/native frontend feasibility now passes numerical
  probes: 105 warmed float outputs agree with original windows within5.92e-5;
  quiet-calibrated INT8 cosine mean/min.9860/.9676 on training-only fixtures.
  All114x96 quiet-LUT P4 output bytes match reference. Exact exhaustive
  activation folding reduces cached inference from~79ms to~70ms. Native FFT
  frontend matches all8,448 quantized source values in33 cases at2.813ms median.
  Each probe restores/verifies the original development app/service/HiESP.
  These are standalone resources/fidelity results, not wake acceptance.
- Sixbit weight-grid compression fails its fidelity gate and newer ESP-NN
  doubles cached inference time despite exact bytes; both are rejected.
  Public reusable LUT/native-mel helpers reproduce the private measured artifact
  and frontend goldens. See `../embedding/README.md` and
  `../results/embedding-cached-p4-v3.json` for methods, hashes and limits.
- The frozen-backbone head recipe is recorded before classifier scores:
  16x96 histories,Conv24/Pool14/Dense16/sigmoid,10k steps,four checkpoints,
  fixed warmed-history and80ms event policy. All23,208 TTS variants are
  prepared; existing disjoint continuous training/validation features continue
  extracting. Training/calibration launches only after both completion markers.
  No fresh testing has been scored; whole-app flash/scheduling fit remains open.
- Follow-up model/arena placement probes match all114x96 bytes in each case:
  flash/internal70.41ms,PSRAM/internal69.85ms,PSRAM/PSRAM70.36ms medians.
  Model expansion into PSRAM does not invalidate the standalone timing result.
  P4 is360MHz/PSRAM200MHz; variable arena16KiB,tensor arena96KiB with76,604
  used bytes. Original app/service/HiESP restores; whole-app fit remains untested.
- Frozen-backbone Conv24 head training and all four INT8 calibrations complete.
  Best strict point is7.5k at.97:recall36.96%,macro41.16%,zero validation
  continuous events. Its continuous-only diagnostic at.6 reaches macro85.64%
  but3events/~7.304h and10.36% maximum confusing-corpus false clips. Reject:
  no paired regressions, reserved scores or full-app candidate flash were used.
  All23,208TTS variants and382continuous sources are complete and hash checked;
  public data/export helpers reproduce sampled feature/model bytes exactly.
  See `../results/embedding-head-conv24-v3.json` and `embedding-head-data-v3.json`.
- DNN32/64 frozen-backbone training and all INT8 validations complete. Strict
  selected recall32.94%/16.56% is unacceptable. A separately frozen continuous-
  FAR diagnostic selects64-wide7.5k at.25; it scores41/50 paired versus HiESP
  48/50,2/100 ordinary-sentence false clips versus0, English3/5.549h and
  Mandarin3/1.294h versus0/0. No reserved scores or complete-app candidate flash.
- Training-only mining selects5,238 difficult windows. Matched same-init10k
  fine-tunes verify identical positives/random draws/PTQ representatives.
  Control41/50 versus mined35/50 paired recall, English false events6 versus1,
  Mandarin2 versus2. Reject both; mining trades recall for fewer false events.
  The reusable miner, explicit warm-start/mining identities and optional
  continuous selection policy are recorded with sanitized result aggregates in
  `../results/embedding-head-dnn-v3.json`.
- A new128-wide40k-step comparison is frozen and running: full loss versus
  upstream-style confidence-filtered loss, same random initialization/data/
  schedule. Empty-active-batch loss/gradient and legacy unfiltered math checks
  pass. No new model recognition, reserved testing or P4 acceptance yet.
- Both128-wide runs and frozen development regressions complete: full36/50,
  confidence39/50 versus HiESP48/50; English false events1/1,Mandarin3/2
  versus0/0. Reject both. See `../results/embedding-head-dnn128-v3.json`.
- Training-only real-environment/room/EQ augmentation passes24 exact replay
  cases over all six TTS engines and starts20,140 added views. Original views
  and all validation remain immutable. Extra-feature loader checks reject
  held-out rows, modified indexes and invalid paths; baseline512 PTQ histories
  must remain identical. A matched warm64 recipe is frozen before new model
  scores. This is a data-domain experiment, not a recognition improvement claim.
- All20,140 environmental views and matched training/calibration complete.
  Control reproduces four prior INT8 artifacts exactly. Expanded paired43/50
  versus41/50, ordinary false1/100 versus3/100, English1 versus6 events and
  Mandarin0 versus2. HiESP remains48/50 and0/0/0; reject replacement. The float
  head diagnostic scores42/50 versus INT8 41/50, so head quantization is only
  part of the gap. New reserved scores remain unread and original board stays
  restored. See the environment and head-quantization result JSONs.
- Local Qwen expansion generates960/960 normal-speed clips from64 remaining
  age/gender/accent/texture conditions, with192 positives and no waveform-QA
  rejects. Source formats/hashes and disjointness from existing head sources
  pass. An independent32clip pilot finds uncertain Nicks/Next pronunciation.
  Source labels are preserved. Full phoneme/ASR diagnostics and new feature
  extraction are running; a control/full/screen recipe is frozen before scores.
- All960 phoneme/ASR audits and3,840 extra feature views complete. The frozen
  consistency union keeps88/192 new positives and all768 negatives;104 uncertain
  positives remain intact outside a3,424-row screened view. The32-source pipeline
  pilot preserves92 retained rows exactly and rejects held-out/audit-mismatched
  inputs before output. Three matched same-init10k-step control/full/screen
  trainers finish; calibration is completing. No new reserved scores or board
  candidate flash. See `../results/embedding-qwen-extension-data-v3.json`.

- Qwen extension final calibrations and all nine development regressions complete:
  control/full/screen paired39/43/41 of50 versus HiESP48; English2/2/2 and
  Mandarin1/1/1 false events versus reference0/0. None exceeds the previous
  environmental43/50, English1, Mandarin0 candidate. Reject all three. More
  synthetic conditions alone did not establish improvement. No reserved scores
  read, no candidate installed. Next is a numerical/resource-only P4 head
  diagnostic, with original-app restoration; see the final Qwen result JSON.

- Temporal32 ordered-convolution/BN comparison completes all training, calibration
  and six development regressions. Plain42/50, ordinary0, English1, Mandarin0;
  dropout44/50, ordinary1, English6, Mandarin1. Neither matches HiESP48/50 and
  zero shared false events. Reject replacements; see the temporal result JSON.
- Current-best DNN64 head standalone P4 tests uncover an optimized FC per-channel
  dispatch bug, then a host-single/MCU-double-rounding mismatch. A local
  registration preserves channel scales and explicitly matches host rounding.
  All299cases x3placements now match every byte; median2.87..2.98ms, tensor
  arena7532used bytes. Five device probes restore and verify original app/slot/
  service/HiESP. No shared dependency source changed. See `embedding-head-p4-v3.json`.
- A fixed two-member equal mean (environmental DNN64 and temporal dropout) is
  frozen before combined validation scores. No fitted weights; exact score
  numerator on1/1536 grid. This is a diagnostic hypothesis, not acceptance.

- Equal-mean validation freezes threshold.4, then existing development regressions
  finish46/50 paired, ordinary1/100, English1/5.549h, Mandarin0/1.294h. This
  improves previous43/50 but still differs from HiESP48/50 and zero shared
  false counts. Keep as next research candidate only. Original board restored;
  no reserved scores read or release. Next actual whole-app slot/CPU/heap fit.

- Matched Hi ESP raw/processed research builds now fit the original OTA slot.
  All 2,709 compile units, dependency lock and sdkconfig are compared; the only
  compile-command difference is the raw-input definition on panel_services.
  Four invalid-configuration guards and 23 runtime-tool tests pass. The frozen
  processed/raw/processed physical bracket reuses 15 positives and 10 ordinary
  negatives per arm. The complete processed/raw/processed bracket now yields
  2/15, 8/15, 2/15 positive triggers and 0/10 ordinary negatives each. Processed
  repeats match every trial; raw gains seven clips and loses one. All 75 trials
  rearm, and all three app swaps restore/verify the original app/service/listener.
  This confirms an input-path confound in the old physical comparison, not
  Hey Nyx parity. No model/threshold training change,
  speech speed change, second mic reader or production-board update. See
  `../results/hiesp-input-build-v1.json` and `../results/hiesp-input-acoustic-v1.json`.

- The frozen raw-input Hi ESP / equal-mean Hey Nyx / Hi ESP physical bracket
  completes all 105 trials: positive triggers 10/15, 9/15 and 7/15; ordinary
  false triggers 0/10 each; near-word false triggers 1/10 each. The Hi ESP
  repeat loses three positive outcomes (two correlated Karen forms and one
  Moira form). This variation prevents interpreting the small count differences
  as model parity. Samantha's `Hi ESB` triggers both controls; her `Hey Nick`
  triggers Hey Nyx. All trials rearm, no source-onset-ambiguous/early events,
  and all three swaps restore/verify original app/service/listener; the final
  live check confirms idle/connected/Hi ESP listening. The 40-second Hey Nyx
  idle cadence and capture-balance gate passes, with its nonzero initial RX
  counter and individual >80 ms inference excursions retained. No lifetime
  zero-overflow, exact latency, call/reconnect or recognition acceptance claim.
  See `../results/raw-engine-acoustic-v1.json`. Keep all control misses; do not
  immediately replay the benchmark to obtain a preferred count. Further
  comparison needs an explicit input/environment diagnostic or a new bound
  candidate, not threshold fitting on these clips. No audio-speed, call
  AEC/AGC/FIFO, model or production-board change. GPU snapshot `ce601c4` rejects
  v20/longer smoothing and starts wider v21 training; its progress is recorded
  separately in `../../handoff/GPU_PROGRESS.md` rather than duplicated locally.

- A new independent endpoint diagnostic freezes all 178 standalone positives
  from the existing six-engine training-only sample, retaining 14 command
  sources separately. It pins the already cached phoneme recognizer, software,
  source/role/hash checks and readout before inference. Seven boundary tests
  pass (CTC repeat/blank timing, geometry, alias scope, altered audio and
  held-out hash overlap). It compares original markers with local CTC feature
  support and RMS endpoints without relabelling, wake scoring, playback or any
  board change. See `../embedding/plans/ctc-endpoint-audit-v1.json`; inference
  has not run at this commit. Its output cannot prove a phonetic truth boundary
  or improved wake quality.

- The frozen independent CTC audit completes all 178 sources, without playback
  or a board change. OpenAI marker-to-CTC-support median is 440 ms and 10/32
  markers exceed the original RMS end by >200 ms; other cohorts have no such
  RMS-late cases. CTC spikes remain an imperfect timing diagnostic, not new
  labels. Full training-only accounting verifies 1,959 unique standalone
  sources and 56 RMS-late anchors. The prior file-end-only rule covers 34 and
  misses 22, including a fixed-sample 840 ms RMS delay. The new generic
  preparation plan freezes all four original/environmental views (224 expected
  endpoint overrides), preserving all source/feature bytes and original
  metadata. It must retain original PTQ inputs before any matched fitting.
  See `../results/embedding-ctc-endpoints-v1.json` and
  `../embedding/plans/training-endpoint-overrides-v1.json`. No new detector
  training, held-out audio scoring, threshold change or quality acceptance yet.

- All-training preparation completes: 224 overrides (112 original, 112 extra
  environmental), 56 unique sources, all 3,068 nontraining rows preserved.
  The opt-in sampler consumer keeps original PTQ metadata. On 40,627 real
  training rows, current default reproduces the retained legacy's 20 batches
  per seed exactly; both 512-example PTQ files match historical bytes. Treatment
  preserves negative draws/labels/RNG and original anchors, changing 35/33
  positive histories in those precheck batches. Six sampler and seven CTC
  boundary tests pass. The exact override manifest and same-initialization
  dense/temporal 10k recipes are frozen before fitting in
  `../embedding/plans/all-training-endpoint-continuation-v1.json`; see
  `../results/embedding-training-endpoints-v1.json`. Training has not started
  at this commit; validation/development/physical improvement remains unproven.

- Both all-training endpoint continuations now complete 10k steps and all four
  original-validation checks. Actual exported PTQ examples retain the exact
  control hashes. Dense selects 2.5k/.6 at 92.3214% macro; temporal selects
  2.5k/.2 at 94.4692%; each has 3 continuous events/7.303778h at its own
  selected point. The two selected models and their cached validation score
  hashes are frozen in `head-equal-mean-all-endpoints-v1-plan.json` before
  the single planned equal-mean calibration. No combined score, development
  regression, new physical playback or acceptance at this commit.

- All-training endpoint equal-mean calibration completes at the pre-existing
  .4 threshold: macro recall94.3065% versus retained-control94.6845%, one
  continuous event/7.303778h for both (upper95%.649508/h). Worst confusing
  false fraction13.2143% still fails the quality gate; approximate median
  endpoint-relative latency139.5ms versus130.375ms. Reject this recipe before
  development/final scoring or physical playback. Both completed trainings have
  exact original PTQ files, source-feature inputs and initialization identities.
  The audit established a real eligibility omission, but its fuller bounded
  sampling correction did not improve recognition in this continuation. See
  `../results/embedding-all-endpoint-validation-v1.json`. Original board app
  stays installed; no PCM speed, I2S/AEC/AGC/FIFO or production-board change.
  Human/room generalization, continuous false wakes, precise latency and long
  call/reconnect acceptance remain unresolved; the full goal remains active.

### Compact-model complementarity check

The [frozen equal-mean comparison](plans/microwake-equal-mean-v1.json) pairs
an existing transfer60 and official64 INT8 model on 6,136 cached validation
clips and 7.324 hours of continuous validation. The score timebases match,
and the event counter reproduced all 1,330 original continuous counts.
The equal mean qualifies under the strict validation limits with zero
continuous events and at most 0.36% false clips in any engine/corpus group,
but its engine-macro latched proxy recall is 39.82%. Neither member alone
qualifies under those same limits; the prespecified fallback comparison is
the best member at the 2% false-clip limit and zero events (44.81%).
At that looser limit the equal mean reaches 56.72%, indicating some
complementarity that is insufficient at the strict point. The result misses
the advancement gate, so do not train a student from this pair. The
[full grid](../results/microwake-equal-mean-v1.json) is reused synthetic and
continuous validation only. No speaker, microphone, board, or reserved
evaluation data was used.

A [paired error audit](../results/microwake-error-overlap-v1.json) then used
the fixed zero-event member points and the two reported equal-mean points.
Of the conservative member's 1,050 positive variant hits, 1,036 also hit
with the other model; 12 of its 13 negative hits overlapped too. The five
strict equal-mean errors are three "Hey Mix" and two "Hey Next" variants.
This indicates limited complementary coverage at those thresholds and
shared near-word errors. It does not say whether the synthetic utterances
were pronounced as intended. The [next matched training plan](plans/microwake-negative-focus-v1.json)
therefore varies only the penalty for high-scoring training negatives and
leaves the training input choices intact.

The [matched INT8 screen](../results/microwake-negative-focus-v1.json)
rejects this loss change. Both arms observe zero events in 7.324 hours of
continuous validation at their diagnostic fallback, but the worst
engine/corpus false-clip rate stays at 4.17%, above the 1% gate. The fallback
macro proxy recall falls from 81.82% to 80.27%. The initial focused export
failed because the upstream clone tried to deserialize the training-only
loss. Resetting the compile metadata for export reproduced the first
training checkpoint byte for byte and completed quantized validation.
No hardware or reserved evaluation corpus was used.

The [terminal-sound counterfactual](../results/microwake-terminal-support-v1.json)
probes both retained INT8 artifacts on 141 distinct training WAVs. The
unmodified clips activate 138/137 (control/focus); fading the clip to silence
40 ms before the independent CTC terminal /s/ support reduces activation to
3/0. Cutting 60 ms after the same support retains 134/134, and no unmodified
clip triggers before the early cut point. The result shows dependence on
late acoustic evidence in this cohort, even though the two models still
confuse similar complete words. The crop is a manipulation of a broad tail
region, and CTC support is approximate; neither isolates the /s/ phoneme nor
proves the training label or real-room timing. No PCM speed, original WAV,
training label, threshold, or firmware changed.

The [near-word audit](../results/microwake-near-word-errors-v1.json) re-renders
all 296 previously scored development views of 74 standalone "Hey Mix" and
"Hey Next" sources and verifies every rebuilt feature array against the cache.
All five strict equal-mean false views, and all eleven quiet views of those
same four source WAVs, retain the distinct /m/ or /t/ in the independent
phoneme decode. The two strict "Hey Next" hits are the same speaker/source;
both cached crossings are more than 270 ms after the estimated /t/ support.
Two additional both-member errors from a different "Hey Next" source omit /t/
in both the raw decode and the firing views; its two non-firing views retain
it. The strict errors therefore cannot all be explained by missing consonants
or scoring before the terminal cue. This favors testing cue discrimination
across render conditions rather than assuming longer smoothing will solve
them. Phoneme decoder outputs remain diagnostic and are never used as new
wake labels. These are development results for the Mac models; no reserved
audio, device, or GPU model was scored.

The [whole-phrase training comparison](plans/microwake-near-word-windows-v1.json)
uses 2,010 original standalone near-word WAVs from six engines. The
preparation checks source SHA/splits and covers all 8,040 views; four long
views keep their ordinary sampling. For half of eligible training draws it
moves the sample window to a band just after the phrase's RMS tail, retaining
the complete onset and the same sources, labels, RNG draws and PTQ inputs.
The paired 1,000-step control reproduces the previous checkpoint and INT8
bytes; both new runs have identical representative input hashes. The
treatment reschedules 8,030 draws and changes 7,983 sampled endpoints.

The [INT8 screen](../results/microwake-near-word-windows-v1.json) rejects the
change. At the fixed .97 reference both see 0 events / 7.3238 h of continuous
development audio. "Hey Mix" false views fall 32/136 to 25/136 and "Hey Next"
falls 17/160 to 16/160, but macro proxy recall falls 81.82% to 78.85% and the
worst engine/corpus false fraction is still 4.17%. Neither arm reaches the
1% gate anywhere in the frozen grid. There is no board promotion.

To reproduce, run `near_word_windows.py --features <feature-dir> --frames 335
--output <study>/windows.json`, then use the plan's common arguments and
per-arm overrides with `train_stream.py`, writing `model-control` and
`model-whole_phrase`. Run `evaluate_stream.py` with the plan's validation
arguments and threshold grid, writing `validation-control` and
`validation-whole_phrase`. `compare_near_word_windows.py --plan <plan>
--artifacts <study> --output <new-report>` verifies the matched exports and
summarizes the original validation stream counts. No playback, board change,
or independent final set was used.

The [same-voice ranking experiment](plans/microwake-paired-ranking-v1.json)
keeps the same eight wake/confuser pairs per batch in both arms. It matches
the same fixed voice, corpus, preset/style/rate and render variant, and
omits Qwen descriptions that do not identify a stable voice. Both runs
complete 8,000 pair draws with identical schedule and representative input
hashes. The ranking loss is training only; both exports have the same INT8
input, UINT8 output and operator inventory.

The [results](../results/microwake-paired-ranking-v1.json) do not advance.
At .97, both have 0 continuous development events / 7.3238 h and the same
50 false views / 3,744. The three matched confuser counts are also identical:
Mix 21/136, Next 17/160, Nick 1/172. Macro proxy recall is 73.97% with plain
BCE and 73.48% with ranking. At their lowest observed worst-group false
fraction (2.32% for both), recall is 64.40% and 64.91%; neither reaches the
1% gate. A new rank gain or threshold sweep on these same samples is not
evidence of real-speaker quality.

To replay this comparison, reuse the verified `windows.json` from the
whole-phrase experiment. Run `train_stream.py` with the plan's common
arguments plus `--near-word-windows <windows.json> --paired-near-words
--paired-near-word-gain <gain>` for each arm. Use the original validation-only
grid and directory naming from the whole-phrase experiment with the new arm
names. Run `compare_paired_ranking.py --plan <plan> --artifacts <study>
--output <new-report>` to verify the schedules, PTQ inputs and fixed-reference
counts. No board changes, playback, or independent final audio were used.

### Auxiliary whole-phrase classification

The [frozen comparison](plans/microwake-paired-phrases-v1.json) trains a
four-way head to identify Hey Nyx, Hey Mix, Hey Next and Hey Nick from the
existing encoder. Both arms use the same 8,000 pair draws, weighted binary
loss, optimizer path and representative inputs. The control's INT8 bytes
match the previous paired control exactly. Both exports retain one binary
output and the same input and operator inventory; the extra head exists
only during training.

The [validation result](../results/microwake-paired-phrases-v1.json) rejects
the treatment. At the fixed .97 reference, mean recall across the synthetic
sources rises from 73.97% to 80.48%, but shared development background goes
from zero to one wake over 7.324 hours. Hey Mix increases from 21 to 28 false
views out of 136; Hey Next from 17 to 21 of 160; Hey Nick from 1 to 2 of 172.
Even at .996 the treatment retains the background wake and its worst
engine/corpus false fraction is 3.04%, above the prespecified 1% gate.
This tests joint training as implemented; it does not show that the head's
class predictions could safely act as a second runtime detector. No final
test data, audio playback, microphone capture or board flashing was used.

To reproduce, reuse the verified pair inputs and common arguments above,
use `--paired-phrase-gain 0` and `--paired-phrase-gain .35` in separate
`model-phrase_control` and `model-phrase_aux` directories, and run the
validation-only grid from the plan with `evaluate_stream.py --threshold-grid`.
`compare_paired_ranking.py` verifies the exported interface, paired input
schedule, calibration and validation exposure and generates the report.

The [background wake diagnosis](../results/microwake-ambient-localization-v1.json)
rebuilds the public LibriSpeech stream and matches the original score array
through the offending utterance. The wake starts 1.97 s into “Have you piles?
Why suffer longer…”. Muting the second sentence leaves the first crossing
at the same sample, and muting the first sentence reduces the score to zero.
Replacing all earlier utterances with silence still yields a wake. The six
other “Have you”, “Do you” or “Are you” onsets found in those already scored
LibriSpeech validation streams do not trigger. This points to a specific
connected speech confuser, not a general question-prefix effect or an artifact
of joining the audiobook chapters. It does not isolate a phoneme or establish
how common that acoustic pattern is. The source stays in development
validation; future training must use separate speakers and utterances.

The [connected speech screen](../results/microwake-connected-speech-screen-v1.json)
tests whether dictionary similarity can find those negatives among the 7,424
utterances from the existing 64 LibriSpeech training speakers. It checks
one- to three-word spans against the unstressed CMU phones for Hey Nyx,
accepts up to two edits or three with a terminal S/Z, and excludes exact
aliases. The pinned local Whisper pass aligns 503 spans in 379 utterances
using agreement with the source transcript and adjacent words. It does not
change the sample rate or labels.

Among 367 utterances with a disjoint same-utterance ordinary comparison,
neither model scores any near or ordinary window at or above .5. More than
98% of the pairs tie for each model. Reapplying the same local-context
score clock to the previous validation confuser reproduces its first crossing
at sample 8390240. That control was scored separately and is not in the
training pool. Do not oversample this pool by dictionary distance alone;
the experiment did not find acoustically difficult negatives for these
models. This is a training diagnostic, not a model update or a new FAR test.

To reproduce, fetch `cmudict.dict` from CMUdict commit
`74790861f652b15e4ac49015a90074ad62a27690`; the candidate script checks its
hash. Run `connected_speech_candidates.py` against the `human-negatives96`
manifest/extracted corpus and `features-expanded-human335/human.json`.
Use the offline ASR environment and the existing `assets/asr-tiny-en`
for `align_connected_speech.py`, then score with `score_connected_speech.py`
using the same feature view and the paired phrase experiment exports.

The [direct score scan](../results/microwake-stream-confusers-v1.json) then
replays all 63.085 h from the 155 remaining LibriSpeech training speakers
through both frozen paired-phrase models. It verifies each feature array and
the original speaker partition; none of these streams comes from validation
or final testing. At .97 the control and auxiliary model produce 3 and 4
events, respectively. There are 22 distinct source utterances above .5
across the two models, and four above .97. Local word alignment places the
four near “she ceased”, the spoken year 1665, “idle words”, and “you think
they will follow us?”. These are approximate locations, not phoneme labels.
The four examples do not support prioritizing one phrase from the dictionary.

Run `mine_stream_confusers.py` against `human-negatives-remaining`, the
original `features-expanded-human335/human.json`, and the paired phrase
experiment. It saves source-clock positions without exporting audio. The
positions let the next matched fine-tune compare mined negatives with
ordinary windows from the same speakers, keeping the development and final
sets untouched. This scan is source selection; its counts are not a new
estimate of production false wakes.

The [matched fine-tune](../results/microwake-matched-confusers-v1.json) uses
one ambient slot in every fourth batch for 250 draws from 22 training
utterances. The control uses ordinary windows from the same speakers and
the treatment uses windows ending at the mined score peaks. Both start at
the auxiliary model checkpoint and match draw order, synthetic phrase
pairs, representative input bytes and exported interface. When a hard
window starts before its speaker stream, both arms get equal leading zero
features (two pairs). No audio playback, board test, or final set was used.

At .97, the mined arm reduces false clips from 68 to 58 out of 3,744,
while macro proxy recall moves from 78.36% to 77.46%. Both still have one
wake in 7.324 h. At .994 the mined arm reaches zero background wakes and
70.84% macro recall, but its worst engine/corpus false clip rate is 2.86%,
above the fixed 1% gate. The ordinary arm has no zero-wake point in the
grid. This is evidence that actual detector-triggering speech is useful
for training, but both exports fail promotion.

For reproduction, build the bank with `matched_confuser_windows.py` using
the frozen direct-score report. Use the
[plan](plans/microwake-matched-confusers-v1.json) and
`--matched-confuser-bank <bank> --matched-confuser-arm ordinary|hard` in
the standard streaming training command. Run the frozen validation grid
and verify the outputs with `compare_matched_confusers.py`.

The [output ceiling probe](../results/microwake-matched-confuser-score-ceiling-v1.json)
replays all 6,136 development clips and 35 continuous streams of the mined
model. Its saved probability score arrays reproduce exactly. The worst
Edge corpus has 16 negative views (7 sources) at the maximum representable
probability, including nine “Hey Next” and five “Hey Mix” views. In this
development set the existing probability cannot satisfy the 1% group gate
without suppressing all activations.

The same model's INT8 tensor before sigmoid separates some of the tied
examples. Under the same gate, exact search over the retained validation
scores allows 44.87% macro recall with reconstructed probabilities or
45.02% with mean logits. Removing output saturation helps, but the best
tradeoff still loses most of the useful recall. This is an upper bound on
the recall achievable by thresholding each tested score on this particular
set. It does not justify a firmware score-format change or establish a new
calibration. Reproduce with `probe_score_ceiling.py` using the frozen hard
model, original feature view and `validation-hard` directory.

### Phone cues at the first wake score crossing

The [offline phone probe](../results/microwake-causal-phone-verifier-v1.json)
reconstructs all 97 detector inputs and matches their cached features. It
uses all 41 false clips (21 original sources) and 56 distinct positive
sources that crossed the mined model's fixed .994 development reference.
The pinned Wav2Vec2 phone model decodes up to 2.4 s of PCM ending at the
first crossing. It detects an /m/ in 14 of 15 standalone “Hey Mix” clips
and a /t/ in 10 of 18 standalone “Hey Next” clips at that point.

Requiring the full phone sequence for “Hey Nyx” rejects all 41 false clips
but retains only 5 of 56 positives at the crossing. Another 240 ms raises
that to 6 of 56; even the complete clean source yields only 9 of 56. Some
near-word cues are already available at the crossing, but this phone
readout is unsuitable for a verifier. These synthetic development clips
cannot establish human pronunciation coverage or acoustic latency. No
audio was played, no microphone was used, and no firmware was changed.

To reproduce, use the local model and Python dependencies pinned by
`plans/microwake-near-word-errors-v1.json`. Run
`probe_causal_phone_verifier.py --features <features-expanded-human335>
--validation <validation-hard> --backgrounds <original-render-backgrounds>
--phone-model <phoneme-audit-model> --output <new-report.json>`, using the
matched fine-tune's `validation-hard` directory. The checked-in result
retains aggregate counts only.

### Mining far-field dialogue

The [dialogue scan](../results/microwake-dialogue-confuser-scan-v1.json)
replays the previous ordinary and mined models on 12 audited AMI training
meetings (4.267 h) and the official Mandarin FLEURS train partition
(9.906 h, including 200 ms separators). It verifies the AMI WAV hashes,
participant and recording exclusions against the GPU track's independent
set, and cached FLEURS feature hashes. A separate metadata check confirms
no FLEURS source hash overlaps either reserved split.

At the fixed .97 diagnostic threshold, the mined model reduces AMI
activations from six to one; both models have zero on FLEURS. At .5, AMI
has 41 and 18 high-score groups respectively, including adjacent groups
that the listener's latch counts as one event. This extends the previous
audiobook result to spontaneous dialogue on training-eligible recordings.
It is not an independent false-wake estimate.

The AMI bank has 36 matched pairs after combining peaks within two
seconds. Each control comes from the same recording, at least ten seconds
away from any retained score peak, with the same amount of history. The
two early pairs use identical leading padding in both arms. No Mandarin
or independent final recording is in the bank.

To reproduce, fetch the GPU source-audit commit and run
`mine_dialogue_confusers.py --fleurs-training <official-train-feature-root>
--experiment <microwake-matched-confusers-v1> --output <scan>`. Then run
`dialogue_confuser_windows.py --scan <scan> --output <bank>`. Use the
[frozen continuation plan](plans/microwake-dialogue-confusers-v1.json)
with the same train and validation commands as the matched fine-tune above,
changing the initialization and bank to the pinned files in this plan.
Neither source collection nor replay involved the microphone or speakers.

Both 1,000-step [continuations](../results/microwake-dialogue-confusers-v1.json)
complete with identical AMI draw schedules, TTS pairs, representative inputs
and exported tensor/operator interfaces. At .97, the matched ordinary and
mined arms have 69 and 64 false clips out of 3,744; macro proxy recall is
75.87% and 78.19%, respectively. Both wake once in 7.324 hours of continuous
development validation. At .992, both have zero continuous events and a
2.86% worst engine/corpus false fraction; recall is 71.51% and 72.87%.
The prior mined model's comparable zero-event fallback at .994 had
70.84% recall and the same worst-group error. Its 16 false Edge variants
are exactly the 16 still present at .992 in the new mined arm, including
14 standalone “Hey Next” or “Hey Mix” views. Neither continuation meets
the 1% gate, so no candidate advances to firmware testing. More ordinary
dialogue alone has not resolved those specific confusions.

### Competing phrase probabilities

The [frozen comparison](plans/microwake-phrase-output-v1.json) changes the
output head rather than discarding the extra phrase predictions. The
treatment predicts other/background, Hey Nyx, Hey Mix, Hey Next and Hey
Nick. All positive windows train the Nyx class; the complete paired
near-word windows train their specific classes, while ordinary negative
windows train background. Only the Nyx probability is exported. Initial
non-target biases make their summed exponential one, preserving the
original sigmoid probability; the first 128 actual training windows
verify a maximum difference of 3.58e-7.

Both 1,000-step arms have identical first-16-batch feature/label/weight
hashes, 8,000 same-voice pair draws, representative inputs and development
exposure. The [INT8 result](../results/microwake-phrase-output-v1.json)
rejects promotion. At .97, macro synthetic proxy recall rises from 77.12%
to 80.23%, and continuous development events fall from one to zero over
7.324 h, but false clips increase from 69 to 78 out of 3,744. The worst
engine/corpus false fraction rises from 5.21% to 7.29%. No point in the
frozen grid meets the 1% gate. Increasing the cached-score threshold to
.998 suppresses every positive and negative view, so a stricter threshold
does not rescue this export. Neither result proves real-speaker recall.

The scalar input/output shapes and types remain unchanged. The treatment
grows from 61,712 to 78,512 bytes, replacing LOGISTIC with SOFTMAX plus a
fixed one-hot projection. Its actual output quantization is recorded and
used by evaluation; it differs from the sigmoid's fixed scale. MCU
arithmetic, operator support, resource use and acoustic behavior are
unproven. No board, microphone or speaker was used.

For reproduction, use the plan's exact initializer, feature view, source
hashes and common arguments with `train_stream.py --near-word-windows
<windows.json> --paired-near-words`. Add `--paired-phrase-gain 0` for the
binary control or `--phrase-output` for the treatment. Run the unchanged
validation-only grid and `compare_phrase_output.py --plan <plan>
--artifacts <paired-run> --output <new-report.json>`.

The [handoff asset](../../handoff/assets-phrase-output-v1.json) identifies
the public research archive containing both original models/checkpoints,
training and calibration reports, validation scores, exact initializer and
pairing bounds. Its per-file manifest and public download hash are verified.
It is explicitly unaccepted and contains no room audio or device state.

### Relative phone-hypothesis diagnostic

The [CTC hypothesis comparison](../results/microwake-ctc-confuser-margin-v1.json)
uses the already frozen trigger-conditioned cohort, restricted to standalone
phrases: 54 distinct positive sources and 34 negative views from 15 sources.
It scores Nyx, Mix, Next and Nick phone hypotheses on the clean source,
the last 2.4 seconds available at the first wake crossing, and the same
window with another 240 ms. No threshold is fitted. The canonical readout
uses one dictionary pronunciation; an exploratory secondary readout averages
a fixed family of vowel/nasal variants with equal per-class priors.

Both readouts identify all 34 clean negative views as their intended class.
At the crossing, the canonical readout rejects 32/34 negatives and retains
32/54 positives; the variant readout rejects 33/34 and retains 31/54.
Adding 240 ms retains 35/54 or 32/54 positives, respectively, with unchanged
negative counts. On complete clean positives the respective counts are
42/54 and 41/54. Thus much of the near-word evidence is available in the
existing causal window, but this large recognizer's output cannot serve as
a useful mandatory gate. Its predictions are not human pronunciation truth
or a justification for relabelling source audio.

The original 88 detector inputs are rebuilt and feature-exact. Source
identity includes text and engine as well as waveform hash and variant:
some synthetic aliases share identical WAV bytes but have different render
seeds. The guarded replay reproduces all scores exactly. Two tiny exhaustive
CTC tests verify repeated phones and equal template priors independently
of the large model. This motivates testing phonetic representation learning
on disjoint training data, rather than a runtime phone-string check or an
added fixed listening delay. No training, firmware, microphone or speaker
change was made in this diagnostic.

Reproduce with the pinned phone-model environment from the near-word audit:
`probe_ctc_confuser_margin.py --features <features-expanded-human335>
--cohort <original-causal-phone-report> --backgrounds <original-backgrounds>
--phone-model <phoneme-audit-model> --output <new-report.json>`.
Run `python -m unittest discover -s research/wakeword_p4/training
-p test_ctc_confuser_margin.py` in that same environment for the numerical
checks. Per-view outputs remain outside the checkout; the published summary
binds the exact retained report hash.

### Training-only temporal phone supervision

The [phone-target preparation](../results/microwake-phone-bank-v1.json)
selects 118 same-voice pairs from the existing training split, across five
engines and three near words. It retains the original binary labels. The
teacher processes each exact 337-hop PCM support window and produces 392
phone posteriors; these are averaged over the trailing 30 ms of each of
67 encoder steps. All 236 rendered feature crops match the original cache.
The source and reserved-waveform checks precede preparation and are repeated
by the training loader.

The actual transfer60 graph's gradients confirm 137-frame receptive fields
at stride three for the first, middle and last encoder steps. Three tests
verify teacher-frame endpoints, binary source roles/reserved-source rejection,
and equality with a normal weighted binary optimizer step when the phone
gain is zero. The teacher sees the complete training crop, so its timestamps
are supervision coordinates, not measured causal phone boundaries.

The [frozen 1,000-step comparison](plans/microwake-phone-distillation-v1.json)
uses the same 118 pairs in both arms. A training-only Dense phone head on
the 67 encoder states adds teacher/student KL with gain zero or .01.
Blank supervision has weight .05; nonblank evidence raises this toward one.
The classifier's original weighted binary loss remains in both arms. The
phone head is discarded before export.

The [INT8 result](../results/microwake-phone-distillation-v1.json) rejects
this recipe. At the fixed .97 reference, both arms have 73 false clips out
of 3,744 and one continuous event / 7.324 h. Macro synthetic proxy recall
moves from 81.04% to 80.32%; the worst engine/corpus false fraction moves
from 4.17% to 5.21%. Neither arm has a zero-event point in the frozen grid.
Both have the exact same first 16 training batches, 8,000 pair draws,
representative inputs, scalar INT8 interface, operator inventory and 25,831
inference parameters. This bounded experiment does not reject phonetic
pretraining in general; it provides no candidate for firmware promotion.

Reproduce preparation with `prepare_phone_bank.py --features <features>
--windows <paired-bounds> --backgrounds <original-backgrounds> --phone-model
<pinned-model> --output <bank>` in the pinned phoneme environment. In the
training environment, use the plan's common arguments, initializer and
`train_stream.py --phone-bank <bank> --phone-gain 0|.01`; then run the frozen
validation-only grid and `compare_phone_distillation.py --plan <plan>
--artifacts <paired-run> --output <new-report>`. The three tests run with
`python -m unittest discover -s research/wakeword_p4/training
-p test_phone_distillation.py`. No board, microphone or speaker was used.

The [public handoff](../../handoff/assets-phone-distillation-v1.json) contains
the original 236-window phone bank, both INT8 exports and binary checkpoints,
initializer, pairing bounds and reports. Per-file hashes and the public
archive download are verified. The training-only phone head's own weights
were not retained; the handoff reproduces its training from the supplied
initializer and bank. The archive is explicitly unaccepted.

### Phone-head fit audit

The [audit replay](../results/microwake-phone-fit-audit-v1.json) preserves
the auxiliary head and measures its unaugmented training-bank fit before
and after the original 1,000 steps. The exported INT8 bytes, first 16 batch
hashes, pair schedule, initializer and representative inputs match the
previous phonetic arm exactly. Its existing rejected validation therefore
applies without rescoring.

Weighted phone KL decreases from 5.3174 to 4.2300 (20.45%). Nonblank argmax
agreement is 95/578 (16.44%); blank agreement is 6,364/15,234 (41.77%). This
is weak fit even on the training bank. The 392-way auxiliary layer used the
encoder's small fine-tuning rate; that remains a hypothesis about the cause,
not a demonstrated explanation. Better training fit would not by itself
establish better wake discrimination. The full auxiliary checkpoint is now
retained as `phone-step-1000.weights.h5`, with its identity in the report.
No new validation exposure, board, microphone or speaker use was needed.

### Frozen-encoder phone-head fit

The [head-fit diagnostic](../results/microwake-phone-head-fit-v1.json)
uses a separate Adam optimizer at .003 for 1,000 updates on cached frozen
encoder states. Binary weights and their optimizer remain byte-identical.
The existing joint-training RNG and pair schedule are untouched. A test
verifies fitting on known targets, unchanged binary weights/optimizer,
and continued zero-gain equivalence with the ordinary binary update.
All four phone-supervision tests pass.

Training-bank KL falls from 5.3174 to .7822 (85.29%); blank argmax agreement
reaches 14,606/15,234 (95.88%). Nonblank agreement remains only 71/578
(12.28%), below the preregistered 50% head-fit requirement. This is not a
demonstrated useful phone readout, despite the improved aggregate loss.
It does not prove absence of phonetic information or reject every possible
distillation target. No firmware candidate follows from this diagnostic.

The first joint harness lacked an automatic pause after warmup. By the time
the failed head fit was inspected it had completed 1,000 updates and entered
float validation; it was interrupted before any validation summary, selected
checkpoint or INT8 export. No control arm or INT8 validation was run. Those
partial joint results are not used. The retained plan and executed source
snapshot identify this attempt explicitly.

The diagnostic now has a dedicated `--phone-warmup-only --steps 0` path.
A replay with the [head-fit plan](plans/microwake-phone-head-fit-v1.json)
produces the exact same cache and head hashes, saves the auxiliary checkpoint
and returns before joint training or validation. Use the plan's common
arguments with the existing feature directory, initializer and `--phone-bank`
to reproduce it. Its output contains source/input identities; the matched
comparison helper also checks head/cache/schedule equality and unchanged
binary state when warmup is explicitly planned. No microphone, speaker or
board was used in either run.

The [public supplement](../../handoff/assets-phone-fit-v1.json) retains both
auxiliary checkpoints, raw diagnostics, frozen plans and executed source
snapshots, including the interrupted harness. The public download and every
listed member hash are verified. It depends on the original phone-bank
handoff and remains explicitly unaccepted. These studies stay silent; any
future room playback needs advance notice of its content and duration.

### Conditional phone identity supervision

CTC blank is an alignment marker, not a physical silence/voicing label. The
earlier head-fit counts cover sparse teacher alignment spikes; they must not
be read as all voiced frames. The
[conditional study](plans/microwake-phone-conditional-v1.json) separates
phone identity from blank by normalizing teacher/student probabilities over
the 391 nonblank outputs and weighting KL by the teacher's nonblank mass.
The original binary wake labels and loss, paired source bank and exported
inference graph remain unchanged.

The [frozen-encoder prescreen](../results/microwake-phone-conditional-head-fit-v1.json)
passes the predefined fit gate. After 1,000 head-only updates, conditional
KL decreases from 5.1614 to 1.1523 (77.68%) and nonblank top-1 agreement is
368/578 (63.67%). Original binary weights and their optimizer are unchanged.
This is training-bank fit, not recognition acceptance. Five tests verify
the original and conditional losses against numerical references, blank
logit independence, zero loss for an all-blank target, finite gradients,
source isolation/timing, frozen warmup and zero-gain optimizer equivalence.
All training launches use the existing immutable source snapshot wrapper.

The [INT8 comparison](../results/microwake-phone-conditional-v1.json) rejects
the .01-gain recipe. At .97, macro synthetic proxy recall moves from
81.04% to 81.25%, false near-word views from 73 to 75/3,744, and the worst
engine/corpus fraction from 4.17% to 5.21%. Each arm has one continuous
event / 7.324 h; neither has a zero-event grid point. Both retain 25,831
inference parameters, the same scalar interface, first 16 training batches,
pair schedule and PTQ inputs. The zero-gain control INT8 bytes match the
earlier original recipe exactly, so its existing validation is reused.

The [training-bank gradient audit](../results/microwake-phone-gradient-audit-v1.json)
checks all 236 crops in 15 batches without updating weights. At gain .01,
phone-to-binary encoder gradient norm ratios range from .00031 to 39.91
(median .0637); the large range reflects almost-zero binary loss in easy
batches and very large binary gradients in hard batches. Gradient cosine
ranges from -.115 to .163 (median .041). The auxiliary signal is therefore
neither uniformly negligible nor consistently aligned with binary learning.
These paired-only batches do not represent the full mixed training batches.
One bounded .1-gain continuation is planned to test the interaction; these
gradient statistics alone do not predict an improvement.

The [single .1-gain comparison](../results/microwake-phone-conditional-gain-v1.json)
also fails. At .97, proxy recall is 81.02%, false near-word views are
71/3,744, the worst group fraction is 5.21%, and the continuous event remains
one / 7.324 h. The auxiliary head reaches 395/578 nonblank agreement (68.34%)
and conditional KL 1.0679, but this better training fit does not establish
better wake recognition. The original control and its validation are reused
byte-exactly; warmup, source schedule, input prefix, PTQ inputs and scalar
interface remain matched. Neither conditional recipe is promoted. No board,
microphone or speaker was used.

The [spike-retention audit](../results/microwake-phone-spike-retention-v1.json)
also shows why the existing averaged bank must not be repurposed as a CTC
transcript. Its collapsed argmax strings average only 2.44 symbols and are
empty for 20/236 rows, including eleven positives. These are teacher
alignment predictions after 30 ms averaging, not verified phone strings.
Any future sequence-level objective must first retain the original teacher
resolution and test its labels; it cannot manufacture transcripts from this
bank or replace the original binary wake labels.

The [public conditional-study handoff](../../handoff/assets-phone-conditional-v1.json)
contains both rejected studies' exact checkpoints, INT8 files, launched
sources, raw validation scores/event traces, frozen plans and diagnostics.
It reuses the original phone-bank/initializer handoff. The public download
and all listed member identities are verified; both studies remain
unaccepted. These results are available to the GPU research branch without
any captured room audio or device configuration.

### Full-resolution teacher sequences

The [retained sequence bank](../results/microwake-phone-sequence-bank-v1.json)
regenerates the same 236 training windows and keeps all 168 original teacher
frames before 30 ms averaging. Features, the original 67-step probabilities,
row order and source/PCM identities match the previous bank exactly.
All full-resolution probabilities are finite, nonnegative and normalized.
Their greedy CTC collapse yields 1,364 phone labels in 236 nonempty strings
of length 3–11, with no special tokens. The longest minimum valid student
path is eleven steps, within the existing 67 outputs. These remain weak
teacher predictions, not human transcripts or new binary wake labels.

The [sequence CTC plan](plans/microwake-phone-ctc-v1.json) uses the same
training-only paired draws and original BCE. It sums over phone alignments
rather than forcing sparse teacher frame timestamps onto the encoder.
The auxiliary loss is CTC NLL divided by target length, with a separate
frozen-encoder head-fit prescreen and the original scalar inference export.
Nine tests cover full-resolution/aligned-target consistency and tampering,
source isolation, repeated phone packing, impossible/empty sequence rejection,
edit counts, CTC agreement with exhaustive paths, finite gradients and
zero-gain binary optimizer equivalence for all three objectives.

The [1,000-update CTC prescreen](../results/microwake-phone-ctc-head-fit-v1.json)
fails the fixed fit gate: normalized CTC loss decreases from 64.2888 to
2.0292, but greedy phone error remains 89.96% with zero exact sequences.
The original binary weights and optimizer remain unchanged. No joint
training, exported candidate or new wake validation follows from this run.
Phone error is still falling between steps 750 and 1,000, motivating one
[5,000-step head-only budget](plans/microwake-phone-ctc-long-fit-v1.json)
with unchanged fit/quality gates. It must reproduce the original head hash
and fit at step 1,000. Observation now records head/draw digests and fit
every 250 updates; loss, optimizer and RNG are unchanged.

The [full-resolution data handoff](../../handoff/assets-phone-sequence-bank-v1.json)
contains the exact bank, source/target audits, frozen plans and source needed
by the GPU work. The public archive and every listed member were downloaded
and hash-verified. It adds no waveforms, room audio, firmware or device state.

The [5,000-update prescreen](../results/microwake-phone-ctc-long-fit-v1.json)
also fails: normalized CTC loss reaches 1.5525, phone error 65.62%, and exact
weak sequences 4/236. The head weights, fit and draw digest at step 1,000
reproduce the original run exactly. Binary weights/optimizer remain unchanged;
no joint training, wake validation or firmware candidate was produced. No
further head-only budget is selected from this curve.

The [fixed beam readout audit](../results/microwake-phone-ctc-readout-v1.json)
helps separate greedy blank preference from the remaining representation/fit
gap. With the same weights, greedy decoding emits 539 phones versus 1,364
teacher labels; beam width 32 emits 880 and reduces phone error from 65.62%
to 50.44%, with only 6/236 exact sequences. A separate exhaustive-path test
checks blank-first remapping, repeated labels and batch order. Greedy totals
reproduce the prior audit, and every model weight remains unchanged. This
does not change the failed fit gate, prove phonetic truth or add a decoder
to the P4. End-to-end encoder learning is a different, still untested
hypothesis; its eventual decision must use wake quality rather than treating
auxiliary transcription as the requested end state.

The [CTC fit handoff](../../handoff/assets-phone-ctc-fit-v1.json) retains both
failed prescreens' raw metadata, auxiliary checkpoints, launched sources,
plans and the complete readout audit. Its public download and all listed
member hashes are verified. The H5 files preserve weights, not head-warmup
Adam slots; the exact longer replay starts from the original seed/bank.
The full-resolution bank is a separate dependency. No source audio, board
configuration, firmware or credentials are included.

### Joint-encoder CTC comparison

The [joint study](plans/microwake-phone-ctc-joint-v1.json) tests a different
hypothesis from the failed frozen-readout studies: whether sequence supervision
can improve the encoder's actual wake discrimination. Both arms replay the
same 5,000-update frozen-head warmup, then train for exactly 3,000 steps with
the same binary learning rate (.0002), batches and schedule. Auxiliary CTC
gain is zero for control and .1 for treatment. Auxiliary transcription remains
diagnostic; previous failed fit gates and reports are unchanged. No further
head-only budget is selected, and the study makes no board change.

The comparison now optionally supports a predeclared control reference when
the binary control has no zero-event grid point. This new plan freezes .97
before training. Treatment must still have zero continuous events, at most
1% worst engine/corpus false fraction, at least three recall points over that
control, and a worst-group false fraction no larger than control's. Without
the explicit plan option, historical comparison behavior remains unchanged.
Three focused tests check opt-in behavior, both false-rate limits, continuous
event rejection, an existing viable control and the recall margin. Passing
only permits further independent evaluation; it does not establish Hi ESP
parity or authorize a firmware update. This study uses no speaker or microphone.

The [training-pair sequence audit](../results/microwake-phone-pair-sequences-v1.json)
checks a limitation of the auxiliary labels independently of wake validation.
One of 118 positive/confuser pairs has exactly the same teacher sequence
(OpenAI TTS, `hey next` stratum); the other pairs differ by one to eight
edits. All 236 sequences reproduce the earlier readout audit exactly. This
teacher target alone cannot distinguish that pair, but the collision is not
proof of a bad recording or binary label. No target, source role, warmup,
joint-training recipe or quality gate is changed by the diagnostic.

The [completed INT8 comparison](../results/microwake-phone-ctc-joint-v1.json)
rejects this recipe. At the frozen .97 reference, macro proxy recall increases
from 81.86% to 84.48% (+2.62 points), but false near-word views increase from
74 to 87 of 3,744 and the worst engine/corpus fraction rises from 6.25% to
7.29%. Both have one continuous event in 7.324 hours; neither has a zero-event
point anywhere in the frozen 21-point grid, including .996. The opt-in fallback
comparison therefore still finds no qualified treatment. No board change or
reserved scoring follows from this result.

Both replayed warmup H5 files exactly match the previous 5,000-step file.
Every warmup history/head/draw/cache identity reproduces the old run. Both
binary models and their .0002 optimizers stay unchanged during that warmup;
the new optimizer is intentionally different from the old .00005 optimizer.
The two joint runs have identical first 16 training batches, 24,000 pair draws,
source schedules, representative inputs and one fixed final checkpoint.
The scalar INT8 interface/operator inventory is unchanged: 25,831 inference
parameters and 61,712 bytes each, with no auxiliary head in the export.

Joint learning does reduce auxiliary phone error: 65.62% after warmup becomes
38.64% in treatment (13/236 exact weak sequences), versus 61.29% in the
binary control (3/236). This establishes that the representation is trainable
on this auxiliary task; it does not establish better wake discrimination.
The launched sources, checks, checkpoints and complete validation results are
retained. Training and validation both used no speaker, microphone or hardware.

The [public joint-study supplement](../../handoff/assets-phone-ctc-joint-v1.json)
preserves both complete runs' checkpoints, executed sources, audits, raw
development scores and event traces, plus the paired-sequence diagnostic.
The public archive download and all 262 listed member hashes are verified.
Teacher-bank and initializer dependencies are separate; full retraining also
requires the original scalar feature cache, which is not in this supplement.
No source audio, room recording, board state, credential or firmware is added.
The recipe remains rejected and the full Hi ESP comparison goal remains open.

The [fixed-reference change audit](../results/microwake-phone-ctc-wake-changes-v1.json)
reconstructs every cached clip maximum/timebase and the original calibrations'
counts without running inference. Treatment gains 66 positive views and loses
35. For negatives, it adds 19 false views and removes six: Piper contributes
nine additions, Edge eight, and Kokoro/OpenAI one each. False source identities
increase from 38 to 42 (seven newly affected, three fully corrected), using
engine/WAV hash/intended phrase rather than treating correlated views or
synthetic voice tokens as independent people. The changed phrase/engine/view
records are retained, including ordinary speech and partial-word errors.

All 35 continuous validation sources are accounted for. Both events occur
at the exact same cached score step, 524.36 seconds into the same public
LibriSpeech stream, within `1183-133256-0018` at 1.94 seconds. Joined PCM,
source FLAC hash and source-start offset reproduce the earlier background
confuser diagnosis. This is a stream timestamp, not a phonetic endpoint or
a new causal localization. The old validation utterance is not added to
training. This broader error pattern does not support attributing the
regression just to the one identical weak teacher pair, or another frozen-head
optimization budget. Model discrimination is still unresolved; no labels,
thresholds, runtime paths, devices or final datasets change in this audit.

Reproduce with `audit_phone_wake_changes.py --artifacts <joint-study>
--features <original-scalar-features> --result
research/wakeword_p4/results/microwake-phone-ctc-joint-v1.json
--output <fresh-report.json>` in the training environment. The complete audit
report is versioned so the GPU branch can inspect every changed view.

### Full mixed-batch gradient diagnostic

The [warmup](plans/microwake-phone-full-gradient-warmup-v1.json) and
[final](plans/microwake-phone-full-gradient-final-v1.json) plans observe the
same original 16 mixed-source training batches at two saved weight states.
They measure binary/auxiliary gradient norms, directions and their shared
global clipping, rather than extrapolating from the earlier paired-crop-only
gradient audit. The optional `--phone-gradient-audit-plan` branch restores
only an explicitly bound auxiliary weight file, verifies that it matches
the selected binary state, computes without optimizer updates, verifies the
original input digest and stops before any training, validation or export.
It does not resume an Adam state or recreate the full training trajectory.

Fifteen phone tests pass, including a closed-form scalar two-task gradient
check, clipping, zero gradients for unrelated heads, immutable weight/optimizer
state and rejection of a changed training prefix. An initial test exposed
TensorFlow's unconnected-gradient handling of Keras dtype strings; the
diagnostic now explicitly represents those unrelated derivatives as zeros.
Existing trainer behavior remains the default; no model recipe is selected
or changed by this observation protocol.

The [completed diagnostic](../results/microwake-phone-full-gradient-v1.json)
reproduces the exact original first-16-batch hash at both fixed states, with
128 pair draws each and no change to full weights or optimizer. Both runs
stop before any training summary, export or validation. The scaled auxiliary
encoder gradient / binary gradient ratio has median .579 (range .077–5.374)
after warmup and .809 (.060–12.713) at the final weights. Encoder directions
oppose on 3/16 and 6/16 batches, with respective median cosine .075 and .043.

Shared global clipping is not a large classifier-suppression mechanism in
these observations: its minimum relative binary scale is .9873 initially,
then 1.0000 at final weights. This does not reconstruct Adam updates or prove
a cause for the wake regression. The variable strength and nearly orthogonal
auxiliary encoder pressure support a bounded task-balancing experiment,
rather than another head-fit budget or an assumption that clipping is the
main problem. Original labels, model recipes, wake thresholds, final datasets,
devices and audio paths remain unchanged by this diagnostic.

### Human pilot preparation

The [paired recording pilot](../human_pilot/README.md) can collect four
counterbalanced pairs per person and condition with an explicit local save.
It preserves the recorded PCM, produces a duration-matched 16 kHz copy,
and reuses the GPU branch's pinned review page and validator to obtain
pronunciation and endpoint ranges without detector scores. Chromium fake
media completed all eight prompts, including the local save. The existing
review parser validated the resulting eight-clip fixture and the native PCM
and replay duration checks passed. The real recording route rejects known
automation captures; the fixture route rejects cross-origin submission and
does not serve audio back. No real audio has been collected or scored at
this snapshot. The pilot does not replace matched tests on the P4.

### Bounded auxiliary encoder experiment

The [balanced CTC plan](plans/microwake-phone-ctc-balanced-v1.json) freezes one
training-only optimization change before either new run: cap the gain-scaled
auxiliary encoder gradient norm at .25 of the binary encoder gradient norm.
Both runs replay the same original 5,000-step head warmup and 3,000-step joint
schedule, with gains zero and .1. Binary and auxiliary head derivatives retain
their definitions before shared global clipping and Adam. The zero-gain path
bypasses balancing. The cap limits gradient norms, not Adam parameter movement;
the logged scalar loss remains BCE+gain*CTC. The inference graph is unchanged.

Eighteen focused phone tests pass, including an actual CTC optimizer update
checked against an independently computed NumPy gradient/clip calculation,
zero-norm cases, original binary-only optimizer equivalence and invalid caps.
The comparison requires finite observed encoder ratios within the frozen bound,
all joint steps accounted for and at least one actual downscaling in treatment.
The plan also binds the old warmup checkpoint and binary control export for
replay auditing. Existing failed studies, validation grid and advancement gates
are retained; no cap sweep or new firmware candidate is authorized by the plan.

This study runs silently from cached features. There is no speaker playback,
microphone recording or board operation. Future audible testing needs advance
notice of its content and duration. The full Hi ESP comparison remains open.

Both frozen runs launched from commit `5c7806e` into
`microwake-phone-ctc-balanced-v1`, using private immutable source snapshots.
The same warmup progress is reproduced through step 1,500 in both live
processes; this is auxiliary training progress, not wake recognition evidence.
The reusable [input auditor](../../handoff/audit_phone_study.py) verifies the
committed plan, executed snapshots, input paths, bank and index hashes,
complete prior warmup replay, preserved binary state, shared batches/draws/PTQ,
the encoder bound and the exact original control export before validation.
It successfully re-audits the completed joint study and rejects a deliberately
changed plan before scoring. Its report explicitly does not claim to hash
the entire external scalar feature cache. Use its `--help` for the explicit
input paths; it accesses no audio device or firmware. Final balanced-study
training and wake validation are still pending at this snapshot.

The [completed balanced comparison](../results/microwake-phone-ctc-balanced-v1.json)
rejects this fixed recipe. At .97, macro proxy recall changes from 81.8621%
to 83.2312% (+1.369 points), while false negative views rise from 74 to 83
of 3,744 and the worst engine/corpus fraction rises from 6.25% to 7.29%.
Both arms have one continuous event in 7.323822 hours and no zero-event point
in the original 21-point grid. Minimum worst-group fractions anywhere in
that grid remain 4.17% and 3.125%, respectively, above the fixed 1% gate.
No cap sweep, reserved scoring, model promotion or board change follows.

The bound was exercised: 2,343/3,000 joint steps downscale the auxiliary
encoder gradient, with mean scale .4888, minimum .002002 and maximum resulting
auxiliary/binary encoder ratio .25000003 (within float tolerance). Control
balancing is disabled. The original control INT8 export, calibration JSON
and complete validation-details JSON reproduce byte-exactly. Both new arms
reproduce the old warmup H5, head/history/draw/cache identities and share
the first 16 batches, all 24,000 pair draws and PTQ inputs. Both validations
exit zero. The scalar export interface/operators, 25,831 inference parameters
and 61,712 model bytes are unchanged; these are not new P4 resource timings.

Auxiliary weak-phone error reaches 44.06%, between binary control's 61.29%
and the previous uncapped treatment's 38.64%. The [cached change audit](../results/microwake-phone-ctc-balanced-wake-changes-v1.json)
reproduces every clip maximum/timebase and all 35 continuous sources. It
finds 62 gained positive views and 29 lost, plus 14 added false views and
five corrected. Added false views are Piper eight, Edge five and OpenAI one.
False source identities increase from 38 to 41 (five added, two removed).
The continuous event again occurs at cached step 17,473, 524.36 seconds into
the public LibriSpeech stream, within `1183-133256-0018` at 1.94 seconds.
This is cached event time, not a phonetic endpoint or a fresh causal test.
The validation utterance is not added to training. Limiting auxiliary pressure
alone has not resolved wake discrimination; training-bank phone fit is not
the acceptance target. All processing remained silent and used no device.

The [public balanced-study supplement](../../handoff/assets-phone-ctc-balanced-v1.json)
retains both runs and their executed sources/checkpoints, postprocessing
commands/scripts, complete development scores/event arrays and supporting
gradient/error diagnostics. Its unauthenticated public download and all 279
listed member hashes are verified. Bank/initializer dependencies are explicit;
the complete original scalar feature cache remains separate. No source audio,
device state, credential or firmware is included. This is a rejected research
recipe, and recognition, real-speaker/room generalization, precise P4 latency,
resources and long call/reconnect parity remain unproven. The full goal stays
open; no additional cap or threshold sweep is selected from these results.

### Compact phrase-sequence readout feasibility

The [new frozen plan](plans/compact-phrase-ctc-readout-v1.json) tests task
compression before another full training run. Instead of 392 weak phone
classes, a Dense(6) CTC head learns the sequence `hey, nyx|mix|next|nick`
from the original synthesis-intent labels of the same 236 training windows.
A temporal-mean Dense(4) readout uses the same frozen [67,60] encoder cache
and identical row draws. The heads have 366 and 244 parameters; their
different losses/structures make this a readout comparison, not an isolated
parameter-count effect. Original binary encoder/classifier weights stay fixed.

Both first get exactly 1,000 training updates. Compact CTC must pass the
predeclared full-bank accuracy, target/confuser, greedy-sequence and loss
criteria before held-engine readout folds can run. A pass in those folds only
supports designing a later open-set experiment. The encoder already saw the
source training corpus, so these folds are not independent acceptance.
Forced-choice phrase scores do not measure silence/ordinary-speech false wakes.
No teacher output replaces a label, and no wake-validation threshold is fitted.

Four tests pass: exhaustive CTC path likelihood versus the TensorFlow result,
word-order sensitivity, alias/label validation, binary confusion accounting,
and rejection of changed cached features or reserved-source overlap. No new
training has launched at this commit. The script is separate from the existing
trainer/runtime and accesses no audio device or board. The source/weights,
bank, index, optimizer, seeds and stop/advance gates are pinned in the plan.

The [completed compact readout result](../results/compact-phrase-ctc-readout-v1.json)
fails that fixed fit gate and stops before all five held-engine folds. Mean
and compact CTC macro accuracy are 93.39% and 93.60%; Nyx recall is 117/118
and 115/118. Confusers assigned Nyx fall from 3 to 1 of 118 fitted rows, but
CTC greedy exact sequence accuracy is only 17/236 (7.20%) and normalized loss
is 1.00034. Next recall is 33/39 for both heads. Forced-choice accuracy does
not establish ordered-sequence learning or open-set false-wake performance.

Both readouts use exactly the same 32,000 row draws and one final checkpoint.
The frozen encoder cache reproduces the earlier cache exactly, and all binary
weights remain unchanged. A read-only input/result audit binds the executed
source snapshot, plan, launch, checkpoints and raw result. Runtime versions
collected after training are explicitly separated from the audit interpreter's
versions; they are not a contemporaneous environment capture. No fold, wake
validation, firmware export or device test follows this failed gate. Do not
extend this head-only budget or loosen gates. The result limits this particular
frozen-encoder recipe; it does not rule out training a temporal encoder.

The [public compact-readout supplement](../../handoff/assets-compact-phrase-ctc-readout-v1.json)
preserves both final checkpoints, full row scores, launched source, plans,
log and input/result audit. The unauthenticated download and all 85 listed
member hashes are verified. The bank/initializer dependencies are explicit;
the complete original scalar feature cache remains separate. No audio,
device state or firmware is included. The full Hi ESP goal remains active.

### Compact phrase CTC encoder-learning diagnostic

The [separate frozen plan](plans/compact-phrase-ctc-encoder-fit-v1.json) now
tests learning the temporal representation itself. It first replays the prior
1,000-step frozen-head control and requires the same final head/cache hashes.
A fresh identical six-token head then trains with the original encoder for
exactly 1,000 updates and the same 32,000 row draws. This changes the trainable
representation and uses a separate .0002 encoder Adam optimizer; head Adam
remains .003. One combined CTC gradient is globally clipped at five before the
disjoint optimizer updates. The unused binary classifier remains fixed.

The prior five fit criteria are unchanged. Only a complete pass permits
designing a subsequent mixed-source binary-wake comparison; there is no
automatic fold, validation, threshold selection, export or board operation.
Fitting these 236 previously seen, correlated training windows can overfit
and is not independent recognition or false-wake evidence. No training has
launched at this snapshot. Five focused tests cover the existing path-sum and
source/label checks plus an independent first-step Adam/gradient calculation,
shared clipping, encoder/head updates, untouched classifier and read-only
measurement. All processing is silent and uses cached features.

The [completed encoder-learning diagnostic](../results/compact-phrase-ctc-encoder-fit-v1.json)
passes all five unchanged fit criteria. On the 236 fitted training rows,
macro accuracy, Nyx recall and greedy exact sequences are all 100%; none of
the 118 confusers is assigned Nyx. Normalized CTC loss is .002153. This is
training fit on already-seen correlated sources, not recognition acceptance.
All scores/row predictions and other fields of the complete frozen-head
reference replay exactly. Both runs draw the same 32,000 row IDs, and the
original encoder cache reproduces. Encoder weights change as intended;
the disconnected original binary classifier remains fixed. All 1,000 joint
updates finish in about nine seconds, excluding replay, loading and scoring.

The recipe changes representation training and its optimizer/combined clip;
it does not isolate an optimizer effect or establish phonetic correctness.
Unlike a longer head-only run, it supports a separately frozen full
mixed-source wake comparison. There is still no validation/final-set score,
INT8 export, audio-device use or firmware change from this diagnostic.

### Full mixed-source compact-sequence comparison

The [new full-study plan](plans/microwake-compact-phrase-ctc-v1.json) freezes
a binary-only versus compact-sequence auxiliary comparison before launching.
Both start from the original binary initializer, not the overfitted diagnostic
encoder. Both share the same 118 original training pairs, source batches,
eight pair draws per batch, augmentation and representative inputs. A shared
1,000-step frozen-head warmup is followed by 3,000 joint updates at .0002 with
the existing schedule. Auxiliary gain is zero versus .1; the original weighted
binary loss remains primary. No teacher output supplies the compact labels.

The training-only head now derives its vocabulary dimension from its explicit
target tensor. The old 392-phone path is unchanged when selected; the new
opt-in sampler reuses its pair-slot scheduling but verifies the exact cached
training crops and source/index/plan bindings. Seventeen focused tests pass,
including original phone behavior, compact loss versus independent TensorFlow
CTC/BCE, binary pair roles, matched draws, source mismatch rejection and the
unchanged scalar inference graph. No production audio or firmware is modified.

The existing 21-point development grid and strict advancement criteria remain:
zero continuous events, at most 1% worst engine/corpus and engine false views,
and at least three macro-recall points over the specified control comparator.
The original binary control export/input identities must replay. Training-bank
fit alone cannot pass. Both final models/checkpoints and all development
score traces will be retained even on rejection. No study has launched at
this commit; all processing will stay silent and use cached features.
