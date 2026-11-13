# Native P4 official-model initialization with exact-zero PCM

This follows the [artifact/build preparation](official_hey_reference_v1.md).
Both official Hi ESP and Hey Willow now have native P4 evidence of five
create/zero-prime/detect/destroy cycles. This is a compatibility/resource smoke
check with integer zeros, not the new paired recognition experiment. No speech
was generated, played, recorded or scored, and no acceptance corpus was opened.

## Fixed diagnostic inputs and hardware

The fixture preparer froze five mono PCM16/16 kHz streams containing only zeros:
16000, 16001, 48000, 48001 and 16000 samples. The odd lengths exercise partial
final chunks. There are 144002 transferred samples (9.000125 seconds), plus the
probe's existing per-clip zero priming. Baud is fixed at 115200. Both arms use
`DET_MODE_95` and the unmodified model bytes authenticated in the inventory.
No threshold is selected or changed using these inputs.

The USB chip was freshly matched against the retained development identity before
using its LAN address. Serial status revealed the changed DHCP address. The chip
is P4 revision v1.3; the probe uses pinned IDF5.5.4/SR2.6.1, 360 MHz and 32 MB PSRAM
at 200 MHz. A fresh partition-table read identified active `ota_0`, offset
`0xe0000`, size `0x690000` (6881280 bytes). Only this verified app slot was written;
the retained original image contains the expected ELF identity. Device addresses,
MAC and raw logs remain private. The dining-room production panel was excluded.

Resource reporting is opt-in (`BASELINE_REPORT_RESOURCES=ON`, default OFF).
Both instrumented builds linked with zero compiler warnings. Their app sizes are
681360/1002000 bytes respectively. Willow retains a standalone 1 MiB partition
warning (4% remaining); that layout is not flashed and is not the provisioned
6881280-byte slot. The default OFF C branch was separately compiled using the
actual Hi ESP compilation flags and a separate object output, with zero warnings;
this was a compile check, not an additional default build/link or device run.

## Actual results

| Measurement | Hi ESP | Hey Willow |
|---|---:|---:|
| Runtime model name | `wn9_hiesp` | `wn9_heywillow_tts` |
| Chunk samples | 512 | 512 |
| Threshold getter, printed to six decimal places | 0.630000 | 0.644000 |
| Median create time | 15.187 ms | 15.163 ms |
| Mean detect time on zeros | 2.093 ms | 2.109 ms |
| Maximum detect time on zeros | 2.863 ms | 2.892 ms |
| Transferred-input detect calls, excluding prime | 284 | 284 |
| Repeated post-first internal heap delta, before create to after prime | 21976 B | 22148 B |
| Repeated post-first PSRAM heap delta, before create to after prime | 319892 B | 321300 B |

All ten transfers have exact WAV identity, sample count, CRC and
`ceil(samples/chunk)` call counts; no zero-input events occurred. The firmware
checks the actual model sample rate and channel count before accepting each clip.
The getter values are observations of this mode, not matched confidence scales or
calibrated operating points for the future comparison.

Heap snapshots include lazy allocations and the per-clip PCM buffer after priming.
The expanded model-loading phase is recorded separately. Both models have a small
first-cycle internal-free change (316/308 bytes); the next four before-create
snapshots are stable in both internal RAM and PSRAM. All first-cycle measurements
are retained. The source of that initial difference is not established, and four
stable repeats do not establish absence of long-term leaks. Zero-input timing is
not speech compute or whole-app scheduling evidence. Successful factory selection
and these calls do not identify proprietary topology or prove the two models have
the same architecture.

Both guarded runs restored original app ELF
`0a37320a642226dc020a94ccf81dfd290eef3723f0fce38e46d4fe1c100329a8`,
`ota_0`, the unchanged selected voice service, idle voice state and Hi ESP wake
listening. A separate fresh HTTP status after both runs confirmed the same state.
The original Nyx connection was already `connecting` before tests and remained
`connecting`; explicit `--restore-readiness app-only` was used. The reports set
`connection_verified=false`; voice connectivity was not accepted or reconfigured.
The runner's default online restoration still requires Nyx up. Both modes reject
changed app/slot/service, busy voice/OTA and a stopped wake listener in host checks.

The first restoration also included an automated UART readiness observation
(DTR/RTS false; terminal-probe replies and `nyx-status`, as in the existing
[readiness instructions](../../panel_wake_runtime/README.md#development-identity-and-boot-readiness)).
Wait for the flash process to finish before opening that reader. The second run
restored without that additional observation. One prematurely started Willow
attempt exited at HTTP preflight before USB access or flashing; its empty output
directory was retained and the completed run used a fresh directory. These are
test-driver observations, not model recognition failures or evidence of their cause.

## Reproduction and evidence

Generate fixtures outside git, build both explicitly selected instrumented probes,
and run each sequentially with the existing baseline runner's complete verified
board/restore arguments. Let each runner finish restoration before starting the
next. Use `--engine MODEL --require-resources --baud 115200`; app-only restoration
is an explicit option for this already disconnected offline situation.

```sh
python research/wakeword_p4/experiments/official_hey_zero_smoke_v1.py prepare \
  --output "$PRIVATE_ZERO_FIXTURES"
python research/wakeword_p4/experiments/official_hey_zero_smoke_v1.py audit \
  --fixtures "$PRIVATE_ZERO_FIXTURES" --report "$PRIVATE_RUN/report.json" \
  --build-report "$PRIVATE_RESOURCE_BUILD_REPORT" \
  --expected-restored-elf "$ORIGINAL_ELF_SHA" --engine "$MODEL" \
  --output "$PRIVATE_AUDITED_REPORT"
```

The auditor independently decodes each WAV, requires exact zero bytes, authenticates
the probe against the concrete build report, and recomputes CRC/call counts and
restoration identities. The original full records and first-cycle measurements
remain in each public audit report; neither scores nor threshold tuning are added.

Evidence in `research/wakeword_p4/results/`:

- `official-hey-native-zero-summary-v1.json`: both native arms, exact evidence hashes,
  fresh identity/partition check scope, final independent restoration and run notes.
- `official-hiesp-zero-smoke-v1.json`, `official-heywillow-zero-smoke-v1.json`: all ten
  original result/initialization frames and phase-specific heap measurements.
- `official-hey-p4-resource-build-v1.json`: concrete compiled probes, model bytes,
  compiler flags and warnings; `official-hey-resource-off-compile-v1.json` records
  the separate resource-disabled C compilation check.
- `official-hey-resource-host-check-v1.json`: original model byte checks, ten explicit
  layout/identity rejections and seven restoration-field mutations in both modes.

The paired speech protocol and new training arms still need GPU/Mac agreement.
Recognition, false-wake rate, keyword-end latency, speech compute, first-utterance
delivery and long-call/reconnect acceptance remain outstanding; the goal stays active.
