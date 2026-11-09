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

## In progress

- Establish the original Hi ESP baseline on P4 using the standalone
  [offline benchmark](../../wakeword_baseline/README.md). Verify PCM CRC and
  restore the original development app after every run.
- V3 context experiment: official 64-channel layer arguments, 203 input frames,
  20,000 steps, batch 128, negative weight 5, mild feature masking and aligned
  float validation. This changes several factors together and does not isolate
  a causal explanation. Its float validation still has no operating point that
  meets the current per-source confusing-negative constraint. Quantized
  validation is being measured without reading test or Mac regression audio.

Keep each run in a fresh private artifact directory. Record source hashes and
failed results. Choose candidates from validation only; reserve new unseen
voices/conditions and real recordings for subsequent acceptance.
