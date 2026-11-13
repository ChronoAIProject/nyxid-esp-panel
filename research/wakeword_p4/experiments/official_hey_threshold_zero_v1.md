# Official threshold API: frozen native zero-PCM check

GPU accepted the official 13-configuration grid in
[Issue #13](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13#issuecomment-6860987242):
default without setter, then requested `.4, .5, .6, .644, .7, .8, .9, .95, .98,
.99, .999, .9999`. This diagnostic repeats DEFAULT at the end: 14 streams per
engine, Willow first and Hi ESP second. It is a separate exact-zero API check,
not a recognition experiment or a revision of the completed speech pilot.

Each stream is the same 16001-sample, 16 kHz mono PCM16 file of exact integer
zeros. Requests are encoded as exact IEEE binary32 bits. Every clip recreates
the model, applies its explicit DEFAULT/SET action, records the actual getter,
performs the original 63 zero-prime calls, then consumes the entire input with
its final partial chunk padded. All prime return values and complete input
return/event accounting are preserved. A final DEFAULT checks actual reset
behavior rather than assuming the preceding request persisted or was reset.

The preparer freezes source hashes, the protocol and complete manifest before
hardware execution. No speech, speaker, microphone, reserved acceptance or
candidate model is used. Preserve every actual getter, merged point and failure;
never interpret setter/transport errors as recognition misses or use this check
to select a recognition threshold.

## Opt-in implementation and boundaries

`BASELINE_THRESHOLD_CONTROL` defaults OFF. Explicit ON builds use the
`official-threshold-v1` ready handshake and strict DEFAULT/SET wire commands.
The runner requires `--threshold-control` and an explicit request field on every
row. Each request obeys the published setter input range `.4.. .9999`.
Getter observations are separately checked as finite values between zero and
one, with decimal/binary32 agreement. Their range and value are not assumed to
equal the setter's input range or requested value.

The new path records default, immediately-after-setting and after-prime getters.
It captures all 63 prime returns and validates complete input no-detect/channel
verified/positive counts. Positive event frames preserve word index and both
capped input and padded chunk timestamps. An event buffer sized for every input
call uses PSRAM; frames are emitted after input completion. This avoids racing
event output with the host's PCM writes and removes the legacy 64-offset limit
from the complete event stream. The old result prefix explicitly reports whether
it is complete. Zero input alone does not exercise native positive-event emission;
host validation of 70 constructed events is analytic evidence only.

The OFF C branch was preprocessed and separately compiled with actual retained
P4 flags, for both models and resource reporting ON/OFF. Preprocessed code differs
from `10402b9` only in four `ESP_ERROR_CHECK` source-line constants. An initial
strict object-byte-equality check failed because those diagnostic constants move;
the failure and real object hashes are retained. Do not claim default linked
binary byte identity. The extension ON branches also compile separately. Full
new links and native executions are recorded separately when available.

Host checks cover all request commands, illegal binary32 values, explicit mode
handshake, setter failure, getter identity, missing/invalid prime returns,
70-event completeness and partial-chunk clocks. Existing official artifact and
restore-field checks also run with the updated host runner. These checks are not
recognition, FAR, real latency or whole-app acceptance.

## Commands

Use the pinned IDF environment with pyserial. All outputs are private fresh
directories outside git; these commands operate no device:

```sh
python research/wakeword_p4/experiments/official_hey_threshold_zero_v1.py check-host \
  --output "$PRIVATE_HOST_REPORT"
python research/wakeword_p4/experiments/official_hey_threshold_zero_v1.py prepare \
  --output "$PRIVATE_THRESHOLD_FIXTURES"
```

Build isolated Hi ESP / Willow projects with pinned IDF5.5.4 and SR2.6.1,
`-D BASELINE_REPORT_RESOURCES=ON -D BASELINE_THRESHOLD_CONTROL=ON`, selecting
the appropriate official model. Keep the original authenticated model bytes.
Use the [guarded native runner](../../wakeword_baseline/README.md) with its full
explicit verified development-chip, active-slot and original-restore arguments,
plus `--threshold-control --require-resources --engine MODEL --baud 115200`.
Verify the chip, actual partition/slot, original app and idle state again before
writing; exclude the dining-room production panel. Each run must finish original
app restoration and readiness before starting the next. App-only restoration is
explicit for the already-offline development configuration and makes no voice
connectivity claim. Do not use the standalone project's `idf.py flash` layout.

After both runs finish:

```sh
python research/wakeword_p4/experiments/official_hey_threshold_zero_v1.py audit \
  --fixtures "$PRIVATE_THRESHOLD_FIXTURES" --build-report "$PRIVATE_BUILD_REPORT" \
  --willow-report "$PRIVATE_WILLOW_RUN/report.json" \
  --hiesp-report "$PRIVATE_HIESP_RUN/report.json" \
  --expected-restored-elf "$ORIGINAL_ELF_SHA" --output "$PRIVATE_ZERO_AUDIT"
```

The audit authenticates source, fixture and concrete probe identities and checks
all original metadata/events, CRC, samples, calls and restoration fields. It
reports the entire observed request→actual mapping, default before/after and
prime/input event counts. A completed zero check establishes API and accounting
observations only; the paired training/calibration/acceptance goal remains open.
