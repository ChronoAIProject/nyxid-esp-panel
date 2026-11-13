# Native P4 baseline on the existing Bryce development cohort

This is an additional native P4 measurement, separate from the already delivered
classic ESP32 QEMU protocol. It uses all 126 original Kristin/Norman WAVs, which
were already consumed by the retained v30 development replay. No new training,
threshold selection or independent acceptance is implied.

The retained `baseline-3mbaud-probe.bin` has SHA256
`28d9fc3d75b25513d4088d8d64780602cf8e8f43fd2fc2a13dcabd1ab0f43a61`.
It is the previous ESP-SR 2.6.1 native P4 `wn9_hiesp / DET_MODE_95` standalone
UART probe, not a new export or full application. Use UART 1500000 for this
run. Each WAV gets a recreated detector and 63 zero blocks (32256 samples),
then its complete unmodified PCM and final partial zero-padded 512-sample block.
The retained probe reports default getter 0.63. In contrast, the classic ESP32
QEMU task explicitly requests 0.63, reports 0.62496001 and adds no extra prime.
Keep these profiles and results distinct.

Before use, freshly verify the development chip identity, active app/slot,
original restore image and idle service/wake state. Read the actual partition
table rather than borrowing another board's layout. The existing guarded
`research/wakeword_baseline/run.py` owns temporary active-slot flashing and
restoration. Do not write bootloader/partition table/NVS, play speakers, record
microphone audio or operate the dining-room production board. Logs, restore
images and device identifiers stay private.

Set `REPLAY_MANIFEST` to the original `bryce-hiesp-replay-inputs-v1/manifest.json`,
`UART_MANIFEST` to a new JSONL file in that same directory, `P4_PROFILE` to a
fresh profile path, `P4_PROBE` to the retained binary and `V30_REPORT` to the
original `bryce-v30-historical-baseline-v1/report.json`. Prepare before inference
and commit the profile with this script's identity:

```sh
python3 research/wakeword_p4/experiments/bryce_hiesp_p4_baseline_v1.py prepare \
  --manifest "$REPLAY_MANIFEST" --uart-manifest "$UART_MANIFEST" \
  --profile "$P4_PROFILE" --probe "$P4_PROBE" \
  --runner research/wakeword_baseline/run.py --v30-report "$V30_REPORT"
```

Pass the resulting JSONL to the guarded runner with the freshly verified board
arguments documented in `research/wakeword_baseline/README.md`. Run every row,
without `--limit`, using `--baud 1500000`. Do not open a second serial owner or
restart an apparently quiet running process. An error is a failed/partial run;
retain it and let the runner restore the original app.

After all 126 rows and original-app/service/wake restoration complete, use the
same arguments above with `summarize` in place of `prepare`, adding
`--report "$P4_RAW_REPORT" --output "$P4_SUMMARY"`. The auditor rechecks every
original WAV/PCM hash, exact manifest rows, transferred PCM CRC, sample/call
counts, event offsets, threshold and retained-v30 source identity. It joins
saved v30 events without another Nyx inference. It reports own-target recall,
the same 54 confusers, opposite wake phrases, voice/condition/kind and standalone
detector timing. There is no continuous-background exposure or FAR estimate.
Generator token-duration timing is a reference, not human phonetic truth.

These results cannot establish acoustic recall, microphone/AEC behavior,
full-app CPU/memory, long-call/reconnect stability or completion of the goal.
No checkpoint, model policy or production firmware changes follow from this run.
