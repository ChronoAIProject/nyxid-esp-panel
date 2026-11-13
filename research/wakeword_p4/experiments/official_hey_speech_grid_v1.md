# Consumed speech: native official 13-configuration witnesses

This freezes all agreed official Willow configurations on the exact twelve
Samantha/Daniel files already consumed in `10402b9`. It supplies native
same-PCM/getter/event witnesses for another runtime to compare, beyond the
completed zero API and greater-than-64-event transport checks. It neither
changes the paired training study nor selects an operating point.

The point-major order is DEFAULT with no setter, then requested `.4, .5, .6,
.644, .7, .8, .9, .95, .98, .99, .999, .9999`. Each point contains the original
twelve source rows, unchanged and in their original order: four intended Willow
targets and eight cross-target streams. All 156 streams use the original
16 kHz PCM16 WAV bytes, without new TTS, rate/gain/trim changes or source selection.
Each stream recreates the model, saves the exact request/default/actual getter
bits and all 63 prime returns, consumes complete PCM with final-only 512-sample
padding, and saves every raw event and all call counts. No custom smoothing or
refractory is added. The concrete threshold-enabled SR2.6.1/IDF5.5.4 Willow
probe from the completed API check is reused at 921600 baud.

`official_hey_speech_grid_v1.py prepare` authenticates the original manifest and
each complete WAV/PCM, copies files verbatim, and writes the entire manifest
and protocol before replay. It binds helper/runner/C/model-build identities and
all 13 getter identities from the actual native zero certificate. Commit the
protocol and helper before inference. Any failure is retained without changing
the grid, extending sources or rerunning to rescue a result.

Use the guarded `research/wakeword_baseline/run.py` with a freshly verified
development-chip MAC, active partition/slot, original restore binary and app
identity. Keep its full restore arguments, plus `--engine wn9_heywillow_tts
--require-resources --threshold-control --baud 921600`. As in the prior native
diagnostics, the original development app is offline, so restoration is explicitly
`--restore-readiness app-only`. Verify original app/slot/service/idle/listener
and an independent final HTTP status after successful runner exit. Do not use
standalone `idf.py flash`; never operate the dining-room production board.

The auditor checks all source and build identities, request/getter/call/CRC/event
accounting and restoration, then retains every clip under every point. It
compares DEFAULT's offsets to the original native speech pilot, reports all
prime positives and records descriptive target/cross event counts. No desired
recognition count is a pass gate. Background/FAR, human word-end latency,
candidate quality, parity, new-source generalization and independent acceptance
remain untested. These already consumed sources cannot enter training or fresh
development/acceptance. A future QEMU replay must authenticate its actual model,
threshold API, lifecycle and complete outputs; this native result alone does not
certify that runtime.

## Completed native witnesses

All 156 streams completed with 20267 input calls, 9828 retained prime returns
and 24 complete positive-event frames. Every PCM/CRC/sample/call/getter check
passes. A separate saved-UART auditor authenticates all 338 detector frames in
order. The 13 actual getter identities match the earlier native zero certificate
and remain unchanged through priming. DEFAULT's complete offsets equal the
retained native speech pilot.

| Requested configuration | Willow target streams with an event / 4 | Cross-target streams with an event / 8 |
|---|---:|---:|
| DEFAULT, no setter | 4 | 0 |
| .4 | 4 | 2 |
| .5 | 4 | 2 |
| .6 | 4 | 0 |
| .644 | 4 | 0 |
| .7 | 0 | 0 |
| .8 | 0 | 0 |
| .9 | 0 | 0 |
| .95 | 0 | 0 |
| .98 | 0 | 0 |
| .99 | 0 | 0 |
| .999 | 0 | 0 |
| .9999 | 0 | 0 |

At requested .4 and .5 the two cross-target streams are Daniel's bare and
command `Hey Nicks` containers. They are preserved, along with every miss at
the higher requested points. There are zero prime positives. These are event
counts on reused synthetic files, with no background exposure or source holdout;
they neither select a threshold nor establish recognition/FAR/parity.

The guarded runner exits 0 after restoring the original app/slot/service/idle/
Hi ESP listener. Separate final HTTP and serial-owner checks pass. Nyx remains
`connecting`, as before the test, so voice connectivity is not accepted.

[All native points and clips](../results/official-hey-consumed-speech-grid-native-v1.json),
[independent frame audit](../results/official-hey-consumed-speech-grid-frame-audit-v1.json)
and [restoration check](../results/official-hey-consumed-speech-grid-restore-v1.json)
record the actual completed scope. The
[public evidence archive](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/official-hey-consumed-speech-grid-native-v1.tar.gz)
contains complete original WAVs, manifest/protocol, original saved report,
sanitized detector frames, build identity and both auditors. Its README gives
commands to regenerate both summaries from saved evidence without device I/O.
The [anonymous receipt](../results/official-hey-consumed-speech-grid-receipt-v1.json)
records size, SHA256, payload checks and reproduction results. This reproduces
saved evidence; another runtime must still execute its own authenticated model.
