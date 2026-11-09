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

## In progress

- Establish the original Hi ESP baseline on P4 using the standalone
  [offline benchmark](../../wakeword_baseline/README.md). Verify PCM CRC and
  restore the original development app after every run. The full run contains
  350 corrected paired renders and two separately reported, previously used
  Hi ESP-plus-request controls.
- Four- and five-engine equal-sampling runs use identical context203 settings
  and seeds, in separate artifact directories. Five-engine features contain
  39,956 sequences. Do not attribute improvement to adding Mac voices if the
  sampling change has not been controlled separately.
- A sixth TTS source is actively generating 1,288 clips from a saved catalogue
  of 46 English Microsoft Edge presets. Named voice families, including their
  multilingual variants, are split before synthesis. No speed/pitch change is
  applied. An initial network/API smoke passed; full generation, pronunciation
  diagnostics and resulting recognition quality remain in progress.
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
