# GPU detector handoff to the P4

Current cross-machine coordination:
[issue #14](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/14).
The original v30 handoff has now been received and tested in the complete P4
application; see the completed probe below. Earlier availability observations
are dated historical snapshots.

This is the artifact contract for a physical development probe. It does not
accept a model for release or change service permissions. The GPU track's
`teacher60` model and the Mac track's 80 ms embedding have different inputs;
their weights and cached features cannot be substituted for one another.

The retained GPU reference at `c0c4a50` is v19, model SHA-256
`4d4db51422775b400b6a5a101b5c01cc9177c95fa7fb61333795c0ca699f5151`.
Its published reports still reject overall acceptance. A future handoff must
name its own commit and hashes; do not assume this snapshot remains latest.
The later inspected snapshot `ce601c4` rejects v20 and starts wider v21
`teacher120` training without quality or MCU results. The topology below
describes v19's `teacher60`; a wider artifact requires its own explicit
metadata, numeric and resource checks. See [GPU progress](GPU_PROGRESS.md).
Snapshot `c560247` subsequently rejects the wider model's recall/false-event
tradeoff. Its denser validation grid also gives the original v19 weights a
different calibration (.9140625), still failing expanded validation. The .95
reference contract below describes the original calibration; a handoff must
bind the actual chosen report rather than silently mixing those policies.

## Required files

Supply a downloadable archive or a shared directory containing:

- The original exported `hey_nyx.tflite` bytes.
- The matching original `training.json`, including architecture arguments,
  `context_frames`, initialization identity, source hashes and model hash.
- The original quantized validation report that fixes the selected threshold,
  score-window length, event latch/refractory and frontend identity. The v19
  report fixes threshold .95 and a five-score mean; no physical-test tuning.
- A synthetic, non-private exact-PCM fixture and expected **INT8 streaming**
  outputs, with its SHA-256, reset/warmup procedure and pinned host interpreter.
  Float-versus-quantized probabilities are a separate comparison and cannot
  establish host-versus-MCU numeric parity.

Keep original reports intact. Any normalized adapter metadata must preserve
and bind their hashes. Do not include account configuration, provider keys,
NVS/flash backups, private UART or room microphone recordings. Existing
published synthetic sources can be referenced by asset name and hash.

## Known metadata boundary

The existing Mac `prepare_runtime.py` expects `input_frames`, a calibration
`selected.threshold`, and reviewed architecture names `transfer60`,
`official64` or `wide128`. GPU v19 instead reports `context_frames: 150`,
`architecture: teacher60`, and the threshold in the quantized validation
report. Its arguments are:

```text
stride 3; first conv 30 filters, kernel 5;
pointwise filters 60,60,60,60;
repeat_in_block 1,1,1,1;
mixconv kernels [5],[9],[13],[21]; no residual connections.
```

The GPU report's validation phase is 1. This is a float-window/streaming
alignment fact, not permission to shift live PCM, duplicate frames or change
speech speed. Importing metadata must explicitly validate this topology and
frontend, preserve the original architecture name, and reject conflicting
model/training/calibration hashes. Renaming it `transfer60` to bypass the
existing allowlist would discard the distinction that needs checking.

## Physical probe sequence

1. Verify the downloaded model and report hashes before loading them. Inspect
   actual TFLite tensor types/shapes, quantization and operators; JSON metadata
   alone does not prove runtime compatibility.
2. Compare the actual 16 kHz, 30 ms window / 10 ms hop, 40-channel MicroFrontend
   configuration and scaling with the existing adapter. The 80 ms native FFT
   embedding frontend is unrelated.
3. Build and run the exact-PCM standalone P4 numerical probe on the explicitly
   identified USB development board, with an available original-app restore
   binary. Compare quantized output bytes and state resets before acoustic
   testing. In particular, do not infer fully-connected per-channel/rounding
   correctness from a float export probe.
4. Check full-app flash, internal memory, inference cadence and hardware audio
   byte balance. Retain the single existing I2S producer, processed call/preroll
   AEC/AGC and ordered FIFO. A smaller model file alone does not prove this fit.
5. Freeze an app/model-bound raw-input Hi ESP / candidate / Hi ESP acoustic
   plan before playback. Count every miss, near-word/ordinary false event and
   onset-ambiguous event. Keep source PCM speed and playback conditions fixed.
6. Restore and verify the original app, service and listener after each probe.
   Do not target the dining-room production panel. Human/room, continuous
   negatives, immediate speech, long-call/echo/reconnect and latency evidence
   remain necessary for overall acceptance.

## Observed artifact availability, 2026-11-11

The GPU source/report branch was inspected through `5740561`; its v28 and
ongoing v29 are summarized in [GPU progress](GPU_PROGRESS.md). A read-only
GitHub release inventory at this snapshot lists `research-hey-nyx-20261110`
and firmware releases through v0.6.28. The research prerelease has 18 assets,
including the earlier Mac model/checkpoint, feature, audio and paired-PCM
handoff archives. No separately identified GPU model/report/INT8-fixture
handoff bundle is published in that inventory. This is not an inspection of
every archive member or a claim that the remote GPU filesystem lacks weights.
Do not infer download availability from published model SHA256 values.

Recheck before a future physical GPU probe:

```sh
gh release list --repo ChronoAIProject/nyxid-esp-panel --limit 8 \
  --json tagName,isPrerelease,name
gh release view research-hey-nyx-20261110 \
  --repo ChronoAIProject/nyxid-esp-panel --json assets \
  --jq '.assets | map({name,size,updatedAt})'
```

No weights were reconstructed, renamed as an approved architecture or flashed
from metadata. Local training research can continue independently; none of
the preceding numerical or synthetic reports establishes deployment readiness.

## Frozen v30 intake

`24c93258fbf2ce88e2f7b71d3219c29941dfafd8` contains the original v30
training, expanded calibration and independent negative reports. The later
`2b940fff128ff342f512b53929b606905ea9bb69` compares the official Hi ESP model
against v30 on exactly the same 73 independent negative streams. On
10.745014097 h of source PCM, v30 has 0 observed events (95% upper bound
0.278802079/h) and official Hi ESP has 1 (upper bound 0.441494490/h). Both pass
the fixed negative bound for this distribution. Neither these counts nor
the reused synthetic positive proxy scores establish real-speaker parity.

The existing build helper now accepts this precise `teacher60` export only
with the original report bytes pinned in
[`gpu-v30-runtime-v1.json`](../panel_wake_runtime/plans/gpu-v30-runtime-v1.json).
No generic `teacher60` admission or changes to the detector/audio path are
needed. Fetch the original files from that commit, use the original model
weights, and generate a fixture using `prepare_runtime_selftest.py` on
normal-speed synthetic mono PCM. Pass the original calibration to
`--app-prime-calibration` to include the 56,160 samples of silence consumed
before the v30 listener starts, then replay the unchanged WAV. The build
requires the bound fixture for v30. The development test must compare MCU output bytes,
reset behavior and event steps; the host's convolution and device's
arithmetic mean can differ at a floating-point threshold boundary.

The training, calibration, preregistration and negative result bytes were
verified against the profile. Twelve configuration tests pass, including
rejection of modified reports and insufficient diagnostic outputs. On the
existing local long335 model, the updated fixture generator verifies 2
subgraphs and reproduces all 123 host output bytes over 59520 samples after
reset. The legacy configuration accepts that actual model, calibration and
fixture. The test fixture and model are not v30 substitutes.

At this snapshot the GitHub research prerelease inventory contains no named
GPU/v30 model or fixture asset, and no 54440-byte TFLite file was found under
the local output and research worktree. The upstream branch publishes reports,
but not the v30 model bytes in its tree. The v30 build and MCU runs still
require the original binary; no board, speaker or microphone was accessed.

GPU commit `04dc65fca8e49d2e765ebf1d3804f85f07fea90f` reports the
preregistered independent paired TTS replay at the frozen thresholds.
Intended target clips with detections: v30 42/78, Hi ESP 45/78.
Common negative clips with detections: v30 4/312, Hi ESP 0/312.
The four v30 events came from TTS intending “Hey Mix” (three) or “Hey Nick”
(one).
Neither engine detected the other engine's intended target clips.
These are clip results for synthesis intent; the recordings do not supply
verified pronunciation or keyword endpoints. They show a gap on shared
hard negatives despite both engines passing the separate read-speech gate.
Do not use these clips to tune a threshold or to infer acoustic parity.

The prime-aware self-test was exercised on the USB P4 using a legacy
149-frame model. The MCU matched all 256 expected output bytes and the
single activation at step 226. It stayed in idle listen for the 40-second
resource check, and the original app and Hi ESP listener were restored.
See [the procedure check](../wakeword_p4/results/legacy-app-prime-selftest-v1.json).
This validates the diagnostic procedure; v30 still needs its own test.

Rechecked on 2026-11-12 through GPU `b6ca5a4`: v34 is rejected and v30 stays
the reference. The named research release assets still do not identify a GPU
model/fixture handoff. This inventory check does not inspect every archive
member or access the GPU host's filesystem. The exact v30 binary needed for
the existing intake remains 54,440 bytes with SHA-256
`04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c`;
the common-calibration export is a different binary and must not replace it.

## Original v30 received and P4 probe completed, 2026-11-12

The GPU's `gpu-v30-p4-handoff-v1.tar.gz` was anonymously downloaded from
`research-hey-nyx-20261110`: 55854 bytes, SHA-256
`60000e37f1b01c408f48f6310bfa7c6d0e837fe6d808127aab9de9a88e1e3cb5`.
All four members and all three payload identities match the frozen Mac intake.
No metadata normalization or operating-point change was needed.

The [complete P4 application result](../wakeword_p4/results/gpu-v30-p4-probe-v1.json)
matches all 179 host output bytes and event step 144. Its synthetic fixture
uses normal 16 kHz PCM, 56160 zero prime samples and only tail padding.
All 12 preselected training-only reset traces reproduce; eight vary and four
yield only zero. The first planned Edge source supplies the fixture. The
all-zero legacy Daniel source remains a rejected diagnostic, not an omitted
control or recognition acceptance result.

Both raw-input 120-second idle arms pass capture conservation, with identical
SDK configuration and matching dependency versions/component hashes.
Normalized detector elapsed fractions are 8.4141%/8.7575%; v30 has 69644 fewer
bytes of minimum internal/DMA free memory. No new overflows occur in the
counted intervals; startup counts and the extra diagnostic are explicit
limitations. Both runs restore the original app/service/Hi ESP listener.

The [public evidence manifest](assets-gpu-v30-p4-probe-v1.json) binds all 43
payload members, including original reports/model, complete diagnostic scores,
synthetic fixture, numeric timing and source. A fresh anonymous download
verifies every member; extracted code regenerates the fixture/report
byte-for-byte and both capture audits. Private board state, firmware, raw
UART and microphone audio are excluded. Commands, dependency pins and the
460800-baud USB fallback are in the [runtime README](../panel_wake_runtime/README.md).

The missing-model blocker is resolved. Recognition, acoustic/word-end latency,
room false wakes and active long-call/echo/reconnect comparison remain open;
v30 is not promoted. Post future completed GPU evaluations and original model
archives to issue #14, preserving the existing experiment protocol.
