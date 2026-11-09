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
  features including these corpora are being generated; these are two more
  corpora from existing engines, not two new engines.
- Downloading the official LibriSpeech train-clean-100 source for more varied,
  training-only human negatives. Preparation verifies its published MD5,
  rejects overlap with all existing speaker partitions, screens wake aliases
  and records actual WAV/features/provenance. Data is not training-ready yet.
- LibriSpeech test-clean (CC BY 4.0) passed the official MD5 checksum and has
  recorded SHA-256 `39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23`.
  Transcript alias screening found no exclusions. It provides 5.403 h of public
  speech plus 0.146 h of explicitly counted separators, across 40 speakers.
  Exact PCM and continuous features are ready. Hi ESP is now running this
  longer negative baseline on the development board; original-app restoration
  is required when it finishes. Never open a second owner of its USB port.

Keep each run in a fresh private artifact directory. Record source hashes and
failed results. Choose candidates from validation only; reserve new unseen
voices/conditions and real recordings for subsequent acceptance.
