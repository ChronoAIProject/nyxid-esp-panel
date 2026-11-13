# Official Hi/Hey Muse: native P4 initialization and zero-input result

The same-firmware host retry completed with original session60079 exit0.
Source/protocol were pushed at `13e6596` before the run. Firmware SHA256 is
`04326b8d47050418498a9e41da7273e5bb6b1436a01308de51ede6ea54798d4b`.
It is the exact v3 app/ELF/model from `d5ba2e1`, with only host frame buffering
changed. ESP-IDF5.5.4, ESP-SR2.6.1 and ESP-DL3.3.14 on the development P4 rev1.3.

Actual factory metadata: `wn10_himuse`, `DET_MODE_95`, 16000Hz, mono, chunk512,
one word named `Hi/Hey Muse`. Default threshold bits `3f1b367a` (about .6063)
are unchanged after priming. No threshold setter was called.

| Original zero samples | Detector calls | Prime calls | Positive returns |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 63 | 0 |
| 511 | 1 | 63 | 0 |
| 513 | 2 | 63 | 0 |
| 16000 | 32 | 63 | 0 |
| 16001 | 32 | 63 | 0 |

All five cases separately create, prime, consume the complete input and destroy.
Only each final input chunk is padded. Seven saved UART frames independently
match all original-sample CRCs, 68 input returns, 315 prime returns, getters and
the report. A copied-fixture mutation deleting an actual return from both the
frame and report is rejected; original files are unchanged. No second inference
is represented by the saved-frame audit.

Creation took 69.228–69.578ms. Zero-input calls averaged about 2.1ms per512
samples, max2.267ms. These are standalone zero-input timings, not speech latency
or whole-panel CPU utilization. Every after-destroy snapshot is identical across
these five cases: internal free563883B, PSRAM free33117536B. Full phase snapshots
and temporary probe-buffer sizes remain in the raw frames; they are not a
continuous peak-allocation measurement or a long-run leak test.

Independent HTTP/no-serial-owner checks confirm the exact original app, ota_0,
selected service/agent, idle voice and Hi ESP wake listener were restored.
Nyx remains `connecting`, as before; this is app-only restoration, not a working
voice-session acceptance. No speaker, microphone, new TTS or dining-room board
was used.

## Failures retained

- `d3354d0`/session6874: packed all four raw p1/p2 files and failed to find
  `wn10_data`. `7f955d1` preserves the failure and independent restoration.
  Official P4 preparation selects p2 and renames the two runtime members.
- `0ad8f95`/session2587: correctly prepared model, but linked ESP-DL3.3.15.
  Precompiled WakeNet allocates144B for `dl::Model`; the linked constructor
  initializes its added member at byte144, with target sizeof152. The heap
  assertion and reboot precede any CASE. `e0d0a97` preserves the native failure
  and exact disassembly/target-size evidence. Compatible3.3.14 is pinned only in
  the separate Muse probe; its object size144 is asserted during compilation.
- `d5ba2e1`/session11658: successful creation and first CASE prefix, then host
  JSON parsing of a partial UART line. `b260b49` preserves it. The successful
  retry buffers until a real newline and retains exact raw UART bytes; model,
  app and ELF were not rebuilt or changed.

The P4-prepared model expansion is430762B; the earlier817708B value was the
unprepared two-chip container. All four upstream raw files remain authenticated.
The official preparation helpers, embedded payload and exact P4 member bytes
are bound in the build reports. ESP-DL3.3.14 header matches immutable upstream
`5acd1c9992af4afb0fa80c221a676982b11381b8`.

## Reproduce the saved-evidence check

From the repository root, without hardware or models:

```sh
MUSE_RESULTS=research/wakeword_p4/results
python3 research/wakeword_muse_reference/audit.py audit \
  --frames "$MUSE_RESULTS/official-muse-zero-frames-v3-host-v2.json" \
  --report "$MUSE_RESULTS/official-muse-zero-native-v3-host-v2.json" \
  --protocol "$MUSE_RESULTS/official-muse-zero-plan-v3-host-v2.json" \
  --build-report "$MUSE_RESULTS/official-muse-zero-build-v3-host-v2.json" \
  --output /tmp/muse-zero-reproduced-audit.json
```

The output file must be new. It should match the committed saved audit exactly.
This milestone establishes initialization, the actual interface and complete
zero-stream accounting. Recognition, common-FAR comparisons, phrase latency and
whole-device voice/first-utterance/long-call acceptance remain unproven.
