# Original c161 development-board trial

The original 60,952-byte model was delivered in GPU branch commit `06fe353c` as
`results/4090-c161-mac-development-model-v1.tflite.gz`, alongside its source and
runtime profile. Copy that original artifact; do not retrain or export it again.
Verify the decompressed SHA-256 against the handoff, then use
`research/panel_wake_runtime/prepare_streaming_runtime.py` with a
`panel-streaming-wake-profile-v1` profile: context 309, threshold 0.91796875 and
smoothing 5. The existing runtime retains refractory 67, half-threshold rearm
and 81,600 samples of continuous startup priming. No cross-platform score or
intermediate-value agreement is required.

For the actual build, reuse the isolated local-development SDK configuration and
the normal IDF 5.5.4 build, with heap tracing disabled and the original 16 MiB
partition layout. Save the previous generated model directory before configuring
the new model/profile. Preserve the normal recovery app/ELF and dependency lock.
Check the completed app fits the existing OTA slot and contains the intended
model bytes. No user or developer credentials belong in model artifacts.

Before flashing, match the existing read-only monitor's development identity,
verify no call/OTA activity and inspect the running OTA slot. Separately verify
the USB chip MAC. The trial wrote only the verified running `ota_0` app at
`0xe0000`; that offset must not be assumed for other boards/layouts. NVS,
filesystem, bootloader, partition table and OTA metadata were retained. Opening
this WCH adapter resets the board, including the chip-identity query; classify
these deliberate installation resets separately. Close serial after flashing.

After boot, verify the new app descriptor through HTTP, reconnection with the
board's own local credentials, and advancing capture/detector counters. A single
short muted call then checks connection, advancing outgoing media, hangup and
wake rearm. Save and temporarily zero speaker volume for that check, stop only
the test-owned call, then restore volume and the unmuted listener. The test
caller stays on the Mac; its credentials must never be installed on the board.
This smoke check does not qualify acoustic wake, AEC or the buffered first
utterance.

The [actual trial](../results/mac-c161-development-p4-trial-v1.json) passed
runtime progress and the short call. The user then reported that the third of
three spoken attempts woke the panel; the connected first request was heard
completely and answered normally. This separates an observed sensitivity
problem from the successful first-utterance path. It is one informal user's
trial, not a population recognition estimate, long-call test or proof that all
transient audio stalls are excluded. No new audio was synthesized, played from
the Mac or recorded. Keep the original model's known near-word/voice-coverage
failures visible, and continue actual-use evaluation before any release.

The development image remains installed for this reversible trial. The dining
room production panel was not changed. Existing passive stability observation
continues across the intentional installation boundary; prior thirteen-hour
evidence still describes the preceding image. Hourly Codex checks are already
installed locally; do not create another scheduler when resuming this work.
