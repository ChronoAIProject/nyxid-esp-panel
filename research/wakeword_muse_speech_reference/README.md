# Consumed-PCM Hi/Hey Muse runtime reference

This separate project compiles the unchanged WakeNet baseline UART/complete-PCM
path with the already validated Muse p2 packer and ESP-DL 3.3.14 ABI assertion.
It does not change either original probe or any model bytes. ESP-SR 2.6.1,
ESP-IDF 5.5.4, revision-1 P4 and DET_MODE_95 stay fixed. The prepared P4 Muse
pack is 430,762 bytes, within the original 512-KiB expansion bound.

The planned descriptive run uses all 108 already consumed Semaine diagnostic
clips whose original literals are Hey Muse, Hey Willow and Hey Nicks, in their
original manifest order. Each clip is run twice with independent create/prime/
complete-input/destroy lifecycles. It uses DEFAULT only, retaining actual
getters and every prime/input event; no threshold setter is requested. Original
labels remain unchanged; a separate report interprets Muse as the target.

This checks a closer word-length reference with actual speech, complementing
the completed initialization/zero-input result. One reused synthetic voice and
no background exposure cannot establish FAR, independent generalization or
mature-model parity. No fresh development/acceptance, new synthesis, acoustic
playback, microphone recording or production panel is part of this run.

Build only in a separate private tree, copying this project and its two source
dependencies with their sibling layout intact. Reuse the six locked components
from the completed Muse zero build. Preserve its `dependencies.lock` and
`sdkconfig.defaults`; never flash the build partition table or bootloader.
Publish the actual build/fixture/protocol identities before any native run.
Execution must use a newly observed explicit USB port/MAC, console-to-HTTP
identity, active slot and full backup, with exact restoration and an independent
app/service/agent/wake-listener check afterwards. App-only readiness must not
be reported as voice connectivity.

`prepare.py` authenticates the actual build against the original zero-probe
model, component/ABI inventory and unchanged baseline source. `fixtures.py`
admits exactly the 108 original selected WAVs against the complete original
manifest SHA `2294fe10ba8a7a72174083e4308a3f055b6c14f4f7dc4ba8b59807c229526bc0`,
their container/PCM SHA, samples and format. It preserves every original row.

`run.py freeze --build-report BUILD --fixture-report FIXTURES --protocol PLAN`
records the actual build/fixture/source identities before native execution;
publish PLAN first. `run.py run` requires that same plan and explicit USB/MAC,
HTTP identity, original image/full-slot hash and partition/slot geometry. The
shared transaction verifies full backup and probe readback, then always restores
and verifies the complete original slot, partition table and OTA metadata after
any write attempt. `check_trained_p4_runner_mac_v1.py` covers this transaction's
failure and restoration paths; the optional reducer leaves its default trained
probe behavior unchanged.

The capture uses 921600-baud UART, no `flush`/`tcdrain`, complete board CRC/call
acknowledgements and a bounded, byte-preserving reader. Two fresh runs are
adjacent for each original clip. A final existing same-baud BAUD/READY command
acknowledges that the final detector destruction completed, without adding an
inference. Capture failures are preserved and cause restoration, never an
automatic repeat. `reduce.py` can independently reduce the saved raw UART and
references; it retains repeat disagreements and all events. Full-clip event
presence is only a descriptive statistic, not phonetic wake latency or an
end-to-end first-utterance success measure. `check.py` exercises these paths
with explicitly constructed data and does not measure recognition.

The first native attempt stopped before any CLIP: UART reconfiguration left a
cut boot-log prefix on the initial READY line. Its original log and complete
restoration are retained. The revised reader and reducer synchronize the first
complete READY after that boot prefix, retaining all raw bytes. Later markers
still require column zero; a displaced later marker is rejected. Freeze the
revised source/protocol before the next explicit run; model, PCM, order,
DEFAULT and all outcome checks remain identical.

## Completed native consumed-PCM diagnostic — 2026-11-14

The explicitly revised run, original session `17243`, genuinely exited 0.
Its protocol was published in `3f51c3a` before execution; the same image,
model bytes, original PCM/order and DEFAULT getter bits `3f1b367a` were used.
All 108 clips completed twice with fresh detector lifecycles: 216 executions,
29,990 input detect calls and 13,608 zero-prime calls. Independent reduction of
the saved UART exactly matches the transaction result. Every repeat has the
same event coordinates/results. Worst standalone input detect call: 2,384 us.

| Original literal | Full clips with events, each repeat | At/after generated-prefix support | Before source |
| --- | --- | --- | --- |
| Hey Muse | 36/36 | 32/36 | 0 |
| Hey Willow | 0/36 | 0/36 | 0 |
| Hey Nicks | 4/36 | 4/36 | 0 |

Muse fires on all nine bare targets and all nine targets in each of the three
command carriers. Four bare target detections precede the generated-prefix
support end by at most 20.75 ms. That support is a synthesis allocation, not a
measured phonetic boundary; these must not be described as proved premature
wakes. The four Hey Nicks cross detections comprise two bare and two timer
clips. All variants and speeds, including failures, remain in the saved result.
The second execution of each clip is a repeat, not another independent source.

Full original 6,881,280-byte app slot, partition table and OTA metadata were
restored byte-for-byte. A separate saved-file and fresh HTTP check confirms
firmware 0.6.27, valid/idle ota_0, unchanged voice service/agent/mode/volume/gain/
greeting settings, idle voice, enabled/listening Hi ESP and no serial owner.
Nyx remains connecting to the absent local backend: this is app-only recovery,
not voice connectivity acceptance. The first independent postprocessor used
the nonexistent status field `model`; its error and original status were kept,
then field access was corrected to `word` without rerunning board inference.

Public evidence is under `../wakeword_p4/results/`:

- `official-muse-consumed-speech-hardware-files-v1.json`: payload hashes.
- `official-muse-consumed-speech-hardware-uart-v1.log.gz`: lossless original
  317,380-byte UART capture, including CRLF/boot prefix.
- `official-muse-consumed-speech-hardware-reduction-v1.json`: all 216 cases,
  original rows, getters, prime returns, events, timings and heap observations.
- `official-muse-consumed-speech-hardware-summary-v1.json`: every literal,
  carrier, variant and speed; generated-support-relative event offsets.
- `official-muse-consumed-speech-hardware-restoration-v1.json`: independent
  restoration checks, without account/service identifiers or credentials.

Decompress the UART without newline translation and pass it to `reduce.py`
with `official-muse-consumed-speech-protocol-v2.json` as `--references`; the full
reduction must match the published reduction exactly. The summary source is
`../wakeword_p4/experiments/summarize_muse_consumed_speech_mac_v1.py`.

This provides a working closer-word-length speech reference on the reused
Semaine source. It does not establish background FAR, other-source accuracy,
native512/raw480 policy equivalence, full-panel headroom, acoustic wake delay,
first-utterance delivery or a qualified candidate. No threshold was selected;
fresh development/acceptance and production hardware remain untouched.
