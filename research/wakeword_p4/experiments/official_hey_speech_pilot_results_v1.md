# Native P4 official speech diagnostic results

The [protocol and implementation](official_hey_speech_pilot_v1.md) were frozen
at `29f88b5` before generating or scoring any of these files. Both native P4
arms completed all 12 fixed streams at 115200 baud, sequentially, and each
restored the original app, slot, selected service, idle state and Hi ESP listener.
No clip was retried, replaced or regenerated, and no threshold was selected.

| Intended-text container measurement | Official Hey Willow | Official Hi ESP |
|---|---:|---:|
| Target bare forms with event | 2/2 | 2/2 |
| Target connected-command forms with event | 2/2 | 2/2 |
| Other-phrase streams with event | 0/8 | 0/8 |
| Events in inserted input leading silence | 0 | 0 |
| Streams with multiple events | 0 | 0 |
| Input detect calls, excluding zero prime | 1559 | 1559 |
| Mean input-container detect time | 2.106070 ms | 2.088763 ms |
| Max input-container detect time | 2.893 ms | 2.867 ms |
| DET_MODE_95 getter, six printed decimals | .644000 | .630000 |

There are only two already consumed macOS sources, Samantha and Daniel.
`Hey Nicks` is the documented Nyx TTS alias; `Hi E S P` spells the letters.
The same full waveforms are used by both arms, without playback, microphone
capture, trimming, gain or speed adjustment. All native sources are 22050 Hz;
their exact rational conversion to 16000 Hz changes duration by less than one
output sample. Each arm receives 49.635875 seconds including inserted and native
silence. Compute timings include silence and final chunk padding, not UART time.

All 24 complete input transfers pass WAV/PCM identity, sample count, CRC and
ceil(samples/512) call-count checks. A second check parses the preserved UART
frames and independently recalculates counts/CRC from WAVs. Full per-clip
offsets, misses, timings and initialization/heap frames are retained in the
evidence. The existing probe performs 63 zero-prime calls (32256 samples) per
clip; it discards their return values. Its reported events cover transmitted
input, including inserted leading silence, not those priming calls. Final
partial-chunk event offsets are capped at input length, as in the original probe.

A separate HTTP read after both successful runner exits confirmed the original
app/slot/service, idle voice and OTA states, and active Hi ESP listening. Nyx
was `connecting` before and after; explicit app-only restoration was used.
Neither voice connectivity nor complete-device voice behavior was accepted.
The verified USB development board was used; dining-room production was excluded.

## Evidence and reproduction

- [Compact native results and all per-clip input events](../results/official-hey-native-speech-pilot-v1.json).
- [Fresh anonymous download and reproduction receipt](../results/official-hey-native-speech-pilot-receipt-v1.json).
- [Evidence archive](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/official-hey-native-speech-pilot-v1.tar.gz),
  1076048 bytes, SHA256 `7e579e7fb95fcb452bfff46917caf1cfeac8bbd2948fff2af5c055bc0909f5ea`.

The archive includes all native synthetic AIFFs, complete PCM WAVs, frozen
protocol/source, manifest, concrete build identity, both original saved reports,
sanitized detector frames and the full audited summary. It contains 41 hashed
payload files plus its manifest. No board address, credential, private boot/flash
log, firmware or proprietary model binary is included. The original frozen v30
handoff archive was not replaced.

Fresh anonymous retrieval authenticates every member. Running the included
auditor reconstructs all 12 WAVs from their native source files, verifies all
24 transfers and reproduces the complete saved summary byte for byte. This is
saved-evidence reproduction, not a second native inference or independent test.
The archive's `REPRODUCE.txt` gives the exact command and package versions.

## What remains

This establishes actual native speech detect calls for official Willow in the
small fixed diagnostic. Four intended-text hits are not evidence of robust
recognition, equal FAR, natural-user behavior or parity of a custom model.
There is no common-confuser/continuous-background evaluation, authenticated
keyword-end latency or independent acceptance. No custom Nyx candidate was
scored or promoted. These consumed sources never return to training or fresh
acceptance.

The GPU's separate official-Willow / custom-Willow / matched-custom-Nyx
protocol, target adapter, exact material/schedules, calibration and independent
acceptance remain the main comparison. Its small-pilot fixed gates do not by
themselves define parity with the mature reference. Later P4 resources,
first-utterance delivery, acoustic behavior and long-call/reconnect validation
also remain outstanding; the overall goal stays active.
