# Wake capture ownership during failed-call teardown

## Confirmed defect and source fix

On base `274246f`, the wake task's normal stop branch keeps capture ownership
while `call_task` joins its peer, microphone, speaker and tone workers. An I2S
read, AGC or FIFO error can instead exit the inner capture loop while `wake_call`
is still attached, reaching `product_audio_release()` before those workers join.
Another audio app could then acquire hardware still used by the old workers.

The fix waits for `wake_call` to detach under the existing `wake_lock`, yielding
without holding the lock. `call_task` already detaches after joining its workers.
It adds no reader, buffering, DSP, model, endpoint or playback-speed changes.
This ownership violation is reproduced below; it is not established as the
cause of earlier acoustic echo, and it is not itself evidence of a memory leak.

## Executed evidence

[`check_wake_audio_release_tail_mac_v1.py`](check_wake_audio_release_tail_mac_v1.py)
extracts and compiles the actual unchanged wake-task release tail with pthread
lifecycle shims. The pending-call case forces the worker-join/detach ordering;
the empty case checks ordinary release. With UBSan:

| Source | Empty call | Attached call |
| --- | --- | --- |
| `274246f` | pass | actual exit 1, premature release reproduced |
| patched source SHA `741aeae018833a3c27cd2c9fec7cbbe7c1b8ea03c30fe10b252e902bdb90abb9` | pass | held until detach, exit 0 |

Both runs had empty sanitizer stderr. The initial ASan+UBSan attempt timed out
during process startup without a case result; it is retained privately and is
not an ASan pass. All 21 existing host test suites passed. The modified
`panel_voice.c` also compiled to a real ESP32-P4 RISC-V object under IDF 5.5.4.

The full firmware image **did not build**: the existing calculator's history
`snprintf` at `esp_brookesia_app_calculator.cpp:531` hit
`-Werror=format-truncation`. An earlier configure attempt was rejected because
development-node mode lacked explicit LAN URL parameters. Both failed builds
remain retained. The ordinary rev1/16 MiB build re-resolved floating dependencies;
the committed lock was restored exactly. Its object check is not a pinned
research-model build or hardware qualification.

Hashes, actual case results, host summary, object identity, build failure and
dependency-resolution scope are recorded in
[`mac-audio-ownership-teardown-regression-v1.json`](../results/mac-audio-ownership-teardown-regression-v1.json).

## Follow-up: full ordinary firmware build completed

On 2026-11-15, the calculator history entry was changed from a fixed 32-byte
buffer to an owning `std::string` containing the complete formula and result.
The old buffer could not hold a 31-character result plus delimiters and formula.
No compiler warning was disabled. The change removes that build failure and
avoids truncating longer history entries.

Original session 90269 genuinely exited 0 after a fresh complete IDF 5.5.4
build. Both the calculator and unchanged ownership-fixed `panel_voice.c` compiled
to P4 objects and linked into the full image. The existing source version is
0.6.28; no version bump or release was made. The 6,191,792-byte app fits the
6,881,280-byte OTA slot with 689,488 bytes spare; 16 MiB flash, minimum chip
revision 100 and rollback are confirmed in the actual generated config.

This is ordinary factory-mode compilation with production URL configuration,
not an experimental Hey Nyx model build. The resolver selected the same ordinary
dependency set as the earlier failed build, differing from the research lock;
the complete differences are reported. The committed lock was restored exactly
to SHA256 `cfcb3e10a756c909dccc57b18f833828724254f282c0745f0cec1cfe541faf4c`.
Original failed build logs remain separate from this new build. No board was
flashed, and neither recognition nor hardware acceptance follows from a link
success. The USB board's existing monitoring baseline remains installed.

Actual object, ELF, binary, source and dependency identities are in
[`mac-audio-ownership-full-image-build-v2.json`](../results/mac-audio-ownership-full-image-build-v2.json).
No board was flashed for this fix. Full FreeRTOS/I2S/codec/network behavior,
long-call/reconnect/AEC and physical error recovery remain separate checks.

Reproduce in new, nonexistent output directories:

```sh
python3 research/wakeword_p4/experiments/check_wake_audio_release_tail_mac_v1.py \
  --source-ref 274246f --expect premature-release --sanitizers undefined \
  --output-dir /private/path/audio-release-before
python3 research/wakeword_p4/experiments/check_wake_audio_release_tail_mac_v1.py \
  --expect safe-release --sanitizers undefined \
  --output-dir /private/path/audio-release-after
```

The read-only background observation deployed during this investigation is
documented separately in [`STABILITY.md`](../../../docs/STABILITY.md). It samples
the original firmware, not the unflashed ownership fix, and cannot replace
long-call acceptance.
