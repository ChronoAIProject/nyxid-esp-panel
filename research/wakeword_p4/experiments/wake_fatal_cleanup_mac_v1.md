# Wake-task fatal resource cleanup

While the GPU ran the actual priming diagnosis, inspection of `wake_task`
identified concrete missing cleanup on fatal rearm paths. This does not explain
the observed idle restarts or establish a progressive idle memory leak.

On base56292c82 the initial failure branch frees its resources, but the later
model recreation/priming failure cannot reach the backing `models`/`root`
owners declared inside initial setup. The later AEC recreation failure exits
without destroying the detector, AGC, owners or six audio buffers. These
resources remain allocated after the wake task deletes itself.

The fix keeps the backing owners at task scope and routes the three fatal
paths to one cleanup tail. It destroys the model before releasing its borrowed
names/weights, releases AEC/AGC and buffers, reports `unavailable`, releases the
audio lease once, and exits. Normal listening, model priming, PCM rate, audio
capture, FIFO, call startup and rearm behavior are unchanged. The task still
reports a fatal initialization failure; no automatic restart is added.

The host probe compiles the actual extracted branches, using tracked real host
allocations and shims for the resource APIs. Its four cases are incomplete
initial allocation, failed model recreation, failed priming after recreation,
and failed AEC recreation. Before the fix, retained allocation counts are
0/2/2/10 and the latter three subprocesses genuinely exit1. After the fix all
four exit0 with zero retained allocations and one task exit/audio release.
The unchanged normal-loop ownership regression also passes both empty and
pending-call cases under UBSan. These are scoped error-path regressions, not
execution of ESP-SR/AEC internals or the complete FreeRTOS task.

```sh
python3 research/wakeword_p4/experiments/check_wake_failure_cleanup_mac_v1.py \
  --source-ref 56292c82 --expect retained --output-dir "$WAKE_OLD_EXIT_PROBE"
python3 research/wakeword_p4/experiments/check_wake_failure_cleanup_mac_v1.py \
  --expect released --output-dir "$WAKE_FIXED_EXIT_PROBE"
python3 research/wakeword_p4/experiments/check_wake_audio_release_tail_mac_v1.py \
  --expect safe-release --output-dir "$WAKE_OWNERSHIP_PROBE"
```

The release-tail extractor now locates the end of the normal loop before finding
its last listening-state assignment; the separate fatal tail must not be
mistaken for the normal ownership tail. An initial fixed cleanup probe failed
on its obsolete source selector before generating C; the corrected probe runs
both old and fixed sources, with the firmware change unchanged.

The actual modified `panel_voice.c` compiled with the installed IDF5.5.4 P4
rev1_3 toolchain in normal mode and with the experimental runtime interface
enabled, with no diagnostics. The experimental check is a translation-unit
compile, not a full experimental link or detector qualification. No firmware
was flashed or released; the development-board monitor keeps its original
0.6.27 baseline and the production board is untouched.

The [result and lossless evidence archive](../results/mac-wake-fatal-cleanup-regression-v1.json)
retain the exact extracted C, compiler logs, subprocess outputs and both P4
compile receipts. There is no audio, model binary or device configuration.

## Full normal firmware integration

The existing normal IDF5.5.4 rev1_3/16MB build subsequently rebuilt the modified
`panel_voice.c`, linked the complete ELF and generated the firmware image.
Original session15556 exited0. The0.6.28 app is6,192,256bytes in a6,881,280-byte
OTA slot, leaving689,024bytes; the actual partition-capacity check passed.
All experimental model and research timing settings were off. This was an
incremental full link, not a new dependency build or an experimental-model link.
One Kconfig warning reported an unset `ESP_IDF_VERSION` environment variable;
the changed C translation unit compiled without diagnostics.

The [full-link receipt](../results/mac-wake-cleanup-full-link-v1.json) records
the source/configuration/artifact identities and scope. The binary stays private
and has not been flashed or released. The passive monitor continues observing
the existing0.6.27 development-board baseline. Integration success does not
prove long-call stability or identify the cause of historical restarts.
