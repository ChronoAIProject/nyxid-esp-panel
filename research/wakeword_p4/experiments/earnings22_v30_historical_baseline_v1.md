Historical v30 continuous Earnings22 development baseline v1
==========================================================

The protocol/source were committed/pushed as `1960850` before the first wake
inference on this reserved cohort. Protocol SHA256:
`2453e0989ce5f2d5cfe15f1b75e173cdd82c3e2ace7b0dcbef7f362daf211a3c`.
The historical v30 bytes/calibration and threshold.91796875 are unchanged.
All twelve complete PCM/feature payloads were authenticated before model load;
full raw-score reset replays were identical for every call. Firmware double
sum/divide and host-convolution event lists have no differences.

Measured source exposure is12.2344634028hours, with1negative-intended background
event in call4483733 at source1488.2s. No event was omitted or reclassified.
Nominal event rate is.0817363187/h; nominal Poisson one-sided95% upper is
.3877460222/h. These calculations ignore call/speaker clustering and do not
establish independent/general-population false-wake acceptance. The provided
transcripts had no exact target-alias text, but omissions/errors are unknown.
This is historical development only; Hi ESP/current GPU candidates/finals
have not been scored here. No threshold search or source/audio change.

The complete actual recording duration matters: upstream metadata labels
call4483733 as3589s, but the official pinned LFS object decodes to6280.49s.
The full decoded recording is preserved; no metadata-based truncation occurs.
The12source durations come from verified decoded sample counts, excluding only
the tiny final frontend pad. No intermediate pauses or audio were inserted.

This continuous-background protocol starts the frontend/model at each complete
call boundary and adds no application prime, matching the original full
background semantics. It differs from the separately disclosed56160-sample
application-prime synthetic-positive diagnostic. Timestamp formula uses native
feature availability; it does not claim application480-sample callback time,
hardware latency, acoustic recognition or long-call/reconnect stability.

The immutable preparation manifest says `wake_models_scored=false` at its
creation stage. This separate score report now marks the cohort as consumed
v30 development. Do not keep calling it untouched after reading these results.
It remains excluded from training and final testing.

This archive contains saved UINT8 score outputs, small reports, exact v30
model/calibration and helper/source bytes. It contains no raw/decoded audio or
frontend-array payloads. Public source URLs/provenance and preparation code are
in `earnings22-development-provenance-v1.tar.gz`; original source licenses are
scoped as described there. Local exact preparation is required for replays.

```sh
python source/research/wakeword_p4/experiments/score_earnings22_v30_baseline_v1.py \
  --protocol study/protocol.json --prepared /path/to/exact/mac-preparation \
  --model-dir model --output /fresh/replayed-reference
```

Every source/feature/runtime hash must agree with the fixed preparation. A
different machine's manifest may differ even when PCM matches; preserve its
actual manifest and freeze an explicit cross-runtime profile before scoring,
never overwrite it with the Mac manifest to bypass identity validation.
Use exact PCM hashes for future officialHiESP same-input comparisons. No
microphone/playback, firmware/OTA, AEC/AGC/FIFO or production-board work occurs.
