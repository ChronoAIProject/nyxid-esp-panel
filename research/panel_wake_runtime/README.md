# Full-application Hey Nyx experiment

This adapter runs a separately trained streaming INT8 model in the existing
panel wake task. It is an opt-in research build, not a replacement for the
production Hi ESP detector. Model files and private acoustic recordings stay
outside git.

The application retains its single 16 kHz I2S producer, communication AEC/AGC,
ordered wake lookback/FIFO, 24 kbps Opus, 20 ms packets and normal speech speed.
While idle, the experimental detector reads raw mic samples from that producer:
its MicroFrontend supplies noise reduction and PCAN. The communication AEC can
suppress its wake cues even with a silent reference. A matched development-panel
capture produces a wake at score 0.996 on raw mic, and zero after AEC/AGC; this
is one diagnostic case, not a recall estimate. The processed preroll and call
audio continue through the existing AEC/AGC path. The default Hi ESP detector
keeps its original input. No second microphone reader or recording task is added.
Two matched Marin acoustic wake/request/reply/quiet/hangup/rearm rounds pass
after this input change. The diagnostic model's offline recall/FAR still fails
the replacement target; this is integration evidence only.

```text
one I2S mic ─┬─ AEC ─ AGC ─ ordered lookback / call FIFO ─ Opus
            └─ MicroFrontend ─ Hey Nyx detector (idle only)
```

The adapter owns only feature extraction and inference. It reports the configured phrase
through the existing wake status and panel label. It does not change service
permissions, Bot selection or provider credentials.

## Build

Use the release's ESP-IDF 5.5.4 environment and dependency versions. Set the
following paths to a private model directory and its quantized validation
calibration. The directory must also contain the matching `training.json`.

```sh
idf.py -B build-hey-nyx \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_MODEL="$WAKE_MODEL" \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION="$WAKE_CALIBRATION" build
```

For an already provisioned development profile, include the documented
`NYX_PANEL_NODE_DEV_MODE`, `NYX_PANEL_AGENT_DEV_API_URL` and
`NYX_PANEL_AGENT_DEV_LOGIN_URL` build options. Keep all registry versions pinned
to the normal release and add only the adapter's declared dependencies. Do not
update unrelated camera, audio or UI packages while measuring a detector.

Configuration verifies the model, training report and calibration hashes,
supported topology, input context, frozen threshold and score-average window.
Reviewed calibration windows are three, five or seven model steps; older
calibrations without the field retain five steps. Invalid explicit metadata
fails configuration. Offline comparisons and the firmware must use the same
bound window. Changing it requires a new validation calibration, not tuning
against the acoustic test. The generated
`manifest.json` under `build-hey-nyx/esp-idf/panel_wake_runtime/generated`
records those identities and detector settings. A failed production-quality
calibration is still usable for an explicitly diagnostic research build; a
successful build does not imply recognition acceptance.

For an inference diagnostic, the training checkout's
`research/wakeword_p4/training/prepare_runtime_selftest.py` creates a private
model-bound binary from an existing 16 kHz mono WAV and host INT8 outputs.
It keeps at most the first four seconds, without changing sample speed. Pass
the resulting path as `-DNYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST=...` to include
the optional startup check. Configuration rejects another model's fixture,
truncated data and invalid sample/output counts. The complete firmware then
compares every output byte through its own frontend and inference path. The
diagnostic detector is destroyed before creating the live microphone detector;
test audio never enters a call. A mismatch leaves wake unavailable and is
reported in UART. Omit this option for ordinary acoustic/resource measurements.

## Temporary board test

Use a USB-connected development board with a known MAC, current app ELF hash,
active OTA slot and a verified local recovery binary. Never target the dining
room production board. Keep its existing partition table, bootloader, OTA
metadata, credentials and filesystem. Stop all other UART readers first.

1. Verify the original app is idle, connected and listening for Hi ESP.
2. Run `acoustic_voice_smoke.py --help` with the original app hash and a known
   good Hi ESP + immediate request WAV. It plays through the Mac speaker and
   checks the complete request, one reply, no extra turns during eight seconds
   of quiet, and cleanup of its own call.
3. Obtain the experimental app ELF hash using
   `python -m esptool image_info --version 2 build-hey-nyx/nyxid-esp-panel.bin`.
4. Run `run_full_app.py --help`. Supply both app paths/hashes, the explicitly
   verified MAC/slot/offset/size and a fresh private output directory. After
   `--`, supply the acoustic command using the experimental hash and Hey Nyx
   fixture. The runner checks the MAC, flashes only that app slot, verifies
   the same service selection and Hey Nyx listening, then executes acceptance.
5. The runner records UART during experimental acceptance, closes its reader
   before reflashing, restores the original app on success or ordinary failure,
   and verifies its identity, connection and original wake listener. Keep the
   runner alive through restoration; SIGTERM requests cleanup, while SIGKILL or
   removing power prevents that guarantee. Inspect `report.json` for
   `restored: true`, including after an unsuccessful acceptance command.

Run the outer runner with an ESP-IDF Python that has `esptool` and `pyserial`.
The acoustic command can use a separate environment with `pymongo`. The local
owner adapter reads test state/history and hangs up the test's own call; the
board continues using its own stored Voice key. No provider/Mac key is copied
to firmware. Keep UART, account identifiers and conversation/audio artifacts
private; publish sanitized aggregate findings only.

Compare both engines under fixed speaker placement and playback volume. A new
TTS phrase that fails the Hi ESP control cannot validate the physical setup;
retain and report such failed controls. One successful wake/request/reply is
integration evidence, not a recall estimate. Recognition acceptance additionally
needs matched voices/conditions, continuous negative exposure, human positives,
distance/noise variation, repeated rearming, immediate speech, playback echo,
reconnect/long calls and full-app CPU/RAM/flash measurements.
