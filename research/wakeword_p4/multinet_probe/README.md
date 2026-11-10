# Silent MultiNet7 feasibility probe

This standalone P4 diagnostic tests the official `mn7_en` acoustic model and
configurable command grammar on **existing training-only synthetic PCM**. It
opens no microphone, speakers, network or NVS. It is not a production wake
replacement or evidence of acoustic, continuous-FAR or full-app acceptance.

The frozen protocol takes two source hashes from each engine/label cohort in
the earlier fixed training audit: 24 sources, six TTS engines, balanced labels.
It preserves every original source sample and adds 2.4 s leading and 1 s trailing
silence. Each 512-sample block uses the original 16 kHz clock. The final short
block receives zero padding. No resampling or speed change occurs.

The two fixed grammars are `HEY NICKS` alone, and that target plus `HEY NICK`,
`HEY MIX`, `HEY NEXT`, `HEY NEXUS`. The official G2P constructs phonemes.
Top-ranked command 1 counts as the target; other detections remain separate.
Threshold .9 comes from the pinned model's advertised default. It is not fitted
on these sources. Reset the decoder per source and after every detected/timeout
state. An 8 s timeout bounds each recognition interval. These reset-separated
short sources cannot estimate always-listening false events per hour.

## Prepare and build

Use the existing pinned project `managed_components`. Keep model weights,
recordings, generated headers, build files and UART logs outside git.

```sh
python research/wakeword_p4/prepare_multinet_probe.py \
  --data-root "$WAKE_DATA" \
  --parent-plan research/wakeword_p4/embedding/plans/training-discrimination-audit-v3.json \
  --plan research/wakeword_p4/multinet_probe_plan_v1_retry3.json \
  --model "$PINNED_COMPONENTS/espressif__esp-sr/model/multinet_model/mn7_en" \
  --output "$NEW_PRIVATE_PROBE_PROJECT"
```

From that generated project, activate ESP-IDF 5.5.4, leave its component manager
enabled, and run `idf.py -B build build`. Preparation writes version/hash-bound
local dependency overrides. The first build disabled the manager and failed
because registry component-name aliases were absent; retry1 fixes that build
configuration without changing sources, PCM, weights, grammar or threshold.
The initial failed build performed no inference or hardware write.

The first hardware run then faults during model creation and restores the
original app/service/listener. Sorted packed-file offsets leave both neural
data and the index 2 bytes past a 4-byte boundary. Retry2 preserves each model
file byte and pads its container payload offset to 16-byte alignment. This is
an initialization hypothesis, not an established cause or recognition result.
It also retains panic register/backtrace lines before recovery. Markdown is
not part of the firmware-template identity; executable source/config files are.

Retry2 still faults before inference. Its dump has `MTVAL=0` and a `strlen`
call from formatted `%s/%s/vocab` path construction. Pinned `srmodel_load`
sets the model base path to NULL; the MultiNet decoder still needs filesystem
reads. Retry3 registers a read-only RAM VFS for the original four model files
and the pinned original initialization command list, then sets the base path.
The target grammars still replace that original command list before PCM is fed.
All model/fixture bytes, threshold and test sources stay unchanged. This uses
the pinned loader's non-header setter in a standalone diagnostic; it is not a
new production loader interface or storage solution.

The generated model and fixture blobs use a size prefix and zlib compression.
Firmware expands both in PSRAM and embeds them in the app. Source WAV hashes,
role/partition separation and dependency identities must match before output.
The model list must contain exactly `mn7_en`; firmware requires 512 samples,
16 kHz and complete bounds/count checks. Preparation does not overwrite an
existing output directory.

## Run only on the development panel

Use `run_multinet_probe.py --help` for the explicit required device, active-slot,
original-app, probe and manifest identity arguments. Supply independently
verified hashes and a recovery binary. The runner checks the running original
app, slot, idle voice state, connected service and exact USB chip MAC before
flashing. It writes **only the supplied active app slot**, never the bootloader,
partition table, filesystem or NVS. Never run `idf.py flash` for this probe.
The template partition CSV supplies a build-time size limit only.

The runner records `MN_READY`, all `MN_CLIP` rows, `MN_TIMING` and `MN_DONE`.
Missing/duplicate clip coverage or incomplete inference counts fail execution.
On success, error or normal termination, its `finally` block restores the
original app and verifies the original slot, service and Hi ESP listener.
Do not SIGKILL it or interrupt power during recovery. A completed probe means
execution finished, including misses and false detections; `accepted` stays
false. It does not mean recognition passed.

Even a successful source diagnostic needs independent validation, real
microphone/AFE transfer, a complete storage solution preserving dual OTA and
GUI assets, whole-app CPU/heap checks, immediate speech continuity and long
call/echo/reconnection tests. Standalone size and timing cannot establish those
requirements.

## Measured result: frozen v1 setting is unusable

The retry3 plan was frozen at source commit `92764b9` before actual inference.
The [retained result](../results/multinet-p4-silent-v1.json) now records actual
P4 execution, rather than the earlier source-only resource estimates:

| Fixed command grammar | Positive target detections | Negative target detections | Other-command events |
| --- | --- | --- | --- |
| HEY NICKS | 0/12 | 0/12 | 0 |
| Target plus four near commands | 0/12 | 0/12 | 1, on an OpenAI positive |

All 48 cases and 7,272 inference calls complete with original PCM speed.
The non-target event's command ID/probability were not logged, so its specific
confusion is unknown. There was no known-supported-command positive control.
The frozen .9 setting is unusable, but the experiment cannot determine whether
that is intrinsic phrase behavior or a remaining model/decoder harness issue.
Before considering another quality run, establish that control and retain raw
decoder outputs; do not blindly tune the threshold on these 24 sources.
Zero negative triggers on short reset-separated clips is not continuous FAR.

Creation takes 333,916 us. Measured heap deltas **during creation** are 6,432
internal bytes and 3,078,104 PSRAM bytes. These exclude the already-expanded
2,751,351-byte model container and 3,711,816-byte test fixture container, other
allocations and linked code. Standalone detect-call p50/p95/max are
8,919/17,373/17,471 us for each 512-sample (32 ms audio) block. These are
standalone timings, not full GUI/voice load measurements or wake latency.

Final destruction logs `esp_mn_commands_clear: The mn commands is not
initialized`, after the probe has explicitly freed the command globals.
`MN_DONE` still reports completion; retain this warning and resolve command
ownership before a future firmware experiment. The frozen probe source and
its result have not been rewritten to conceal it.

The automatic runner fails its clip-coverage check after parsing only 46/48
rows. Its old `read_until` handling inserted a newline when a timeout split
`MN_CLIP` after `M`. The retained log contains both JSON payloads intact.
Offline recovery joins exactly original lines 102–103 and 118–119 (arm0,
clips4/19), preserving the original log/report hashes and failed-run status.
Full clip identity/sample coverage and 7,272 calls then pass validation.
There was no hardware rerun for this repair. Reproduce with:

```sh
python3 research/wakeword_p4/audit_multinet_log.py \
  --uart "$PRIVATE_RESULT/uart-private.log" \
  --report "$PRIVATE_RESULT/report.json" \
  --manifest "$PRIVATE_PROJECT/vectors-manifest.json" \
  --expected-uart-sha256 ab2af42833b192951f585024a05cc88ce676dcf5868fa144b4fe016200e2e42a \
  --expected-repairs 2 --output "$NEW_AUDIT_REPORT"
```

The maintained runner now writes exact UART bytes and buffers fragments until
a real newline; it rejects duplicate summaries and unterminated completion
records. Seven host tests cover all stream split boundaries, empty timeout
reads, bounds, incomplete/duplicate/missing results, explicit recovery and
damaged payloads. They perform no hardware or audio operations:

```sh
python3 -m unittest discover -s research/wakeword_p4 -p test_multinet_log.py -v
```

The v1 plan binds the original runner revision. The updated recorder is for
future independently frozen runs; it is not claimed to have recorded this
historical result. After every hardware attempt, including this failed-runner
case, app/slot identity, original service and Hi ESP listening were verified
restored. No production firmware source or dining-room panel changed.
