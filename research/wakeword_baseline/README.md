# Hi ESP offline baseline on a development P4

This standalone app runs pinned ESP-SR 2.6.1 `wn9_hiesp` with `DET_MODE_95`,
matching the production detector. It accepts lossless 16 kHz mono PCM16 over
UART and reports detection sample offsets, inference time and a PCM CRC32.
It does not read the microphone, connect to Wi-Fi or write NVS/flash.
This is an offline detector benchmark, not acoustic or full-app acceptance.

The initial project configuration targets P4 revision <3 and 32 MB PSRAM.
Build with the pinned ESP-IDF 5.5.4. Verify another board's revision, layout and
active app slot before using it. Keep the production dining-room panel out of
these experiments.

```sh
source "$IDF_PATH/export.sh"
idf.py -C research/wakeword_baseline build
```

**Do not run `idf.py flash`.** Its standalone partition layout differs from the
provisioned panel. `run.py` writes only the explicitly verified active app slot.
It checks the USB MAC, original app ELF identity, idle state, slot size and
fixture WAV hashes first. It restores the original app in `finally`, then
checks its app identity, selected local voice service and wake listening.
Normal SIGTERM cancellation also enters restoration. Power loss or SIGKILL
cannot be recovered by a running Python program; retain the original binary
and recovery command independently.

Use a Python environment containing esptool and pyserial; WAV reading uses the
standard library. All output directories and fixture audio must remain outside
git. Arguments below intentionally use placeholders, not a device identity.

```sh
python research/wakeword_baseline/run.py \
  --port "$DEV_PORT" --expected-mac "$DEV_MAC" --status-url "$DEV_STATUS_URL" \
  --restore-app "$ORIGINAL_APP" --expected-app-sha256 "$ORIGINAL_ELF_SHA" \
  --app-offset "$VERIFIED_OFFSET" --slot "$VERIFIED_SLOT" --slot-bytes "$VERIFIED_BYTES" \
  --probe-bin research/wakeword_baseline/build/nyx_wake_baseline.bin \
  --manifest "$FIXTURES/manifest.jsonl" --output "$PRIVATE_RESULTS"
```

Start with `--limit 14` for a transfer/reset smoke check. Console startup is
115200 baud; the benchmark negotiates faster UART only after a readable ready
message. `--baud 115200` retains the slow rate for diagnostics. A truncated or
corrupted transfer is an error, never a counted recognition miss.

`training/baseline_samples.py` in the sibling `wakeword_p4` project generates
paired Hi ESP / Hey Nyx Mac TTS fixtures with identical random augmentation
conditions. Shared ordinary speech and phrase-specific near words are included.
The old Mac fixtures have informed development, so this corpus is a regression,
not a fresh real-user test. Different word lengths mean latency must be measured
from keyword completion, not just from WAV start.

The same offline fixtures can be scored by the host INT8 model. Each newly
selected model also needs a P4 numeric comparison and full-app acoustic tests
before claiming equivalence to Hi ESP. Do not tune thresholds on the final
comparison or report synthetic recall as real-user recall.

`compare.py` verifies the manifest identity, complete result count and app
restoration before exporting aggregates without private WAV paths. With
`--model` and `--calibration`, it also runs a Hey Nyx INT8 export using the
threshold already chosen on a separate validation set. Its optional inference
uses the sibling training environment; baseline-only summary is standard Python.

```sh
python research/wakeword_baseline/compare.py \
  --baseline "$PRIVATE_RESULTS/report.json" --manifest "$FIXTURES/manifest.jsonl" \
  --output "$PRIVATE_RESULTS/summary.json"
```

Bare `ESP` in macOS TTS can expand to the wrong word. Positive fixtures were
regenerated with explicitly spelled `E S P`. The current six ESP near-word
texts have not had the same pronunciation validation, so the summary separates
them from shared negatives. Previously used Hi ESP-plus-request controls are
also separate from matched keyword-only recall. RMS keyword-end estimates can
place a legitimate early WakeNet detection outside a narrow endpoint window;
both any-event and fixed-window measurements are reported, with that limit.
