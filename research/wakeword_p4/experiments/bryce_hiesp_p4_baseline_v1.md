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

## Completed native P4 replay, 2026-11-13

All 126 original WAVs completed on the freshly verified company development P4
revision 1.3. Every full PCM CRC, sample count and 512-sample call count passed;
the 126 saved decoded UART result objects also match the final report exactly.
The original active app, selected voice service, Nyx connection and Hi ESP
listener were restored and independently rechecked through fresh HTTP status.

| Existing synthetic development | Native P4 Hi ESP | Retained host v30 |
| --- | ---: | ---: |
| Own wake phrase, any event | 29/36 | 14/36 |
| Common confuser clips with event | 0/54 | 7/54 |
| Opposite wake phrase clips with event | 0/36 | 0/36 |
| Bare own wake phrase | 14/18 | 11/18 |
| Own wake phrase plus command | 15/18 | 3/18 |
| Norman own wake phrase plus command | 9/9 | 0/9 |

The native detector used 21735 calls, mean 2129.8427 microseconds per 512-sample
block, maximum 2944 microseconds. These are standalone inference timings, not
full-app CPU utilization. Descriptive generator-reference latency medians are
74.4375 ms and 177.59375 ms respectively; they use different detected subsets,
phrases and frontends, so their difference is not a paired speedup estimate.

The gap remains substantial, particularly for connected commands. This gives
another concrete comparison for source/generalization research; it does not
identify a unique cause or justify lowering a threshold. No new Nyx inference,
new training, rescoring of a final set or promotion occurred.

The raw report, all exact PCM, source, retained standalone probe and licenses
are published as `bryce-native-p4-hiesp-baseline-v1.tar.gz` under release
`research-hey-nyx-20261110`. Archive SHA256:
`e8aab82bf0d2d858939fc4f0ea5c59333d81981e99ab1524c38913edd1c17df7`.
The package is 14104942 bytes with 156 payloads and one contents manifest.
Extracted source recreates the complete summary byte-for-byte without model
inference or hardware access. Repository summary:
`results/mac-bryce-native-p4-result-v1.json`; original pre-inference profile:
`results/mac-bryce-native-p4-profile-v1.json`.
