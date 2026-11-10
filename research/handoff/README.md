# Hey Nyx research handoff for an NVIDIA RTX 4090

Snapshot: 2026-11-10. Branch: `research/hey-nyx-4090`.
This branch combines the complete Mac training research and the experimental
full-panel runtime. Production still uses **Hi ESP**. Hey Nyx is **not accepted
as an equivalent replacement**. The dining-room production panel is excluded.

## Start on Linux

```sh
git clone --branch research/hey-nyx-4090 https://github.com/ChronoAIProject/nyxid-esp-panel.git
cd nyxid-esp-panel
uv venv --python 3.12 .local/4090-env
uv pip sync --python .local/4090-env/bin/python research/handoff/requirements-gpu.lock
.local/4090-env/bin/python research/handoff/gpu_smoke.py
python3 research/handoff/fetch_assets.py --output "$PWD/.local/research-snapshot" \
  --include-audio --include-history
export WAKE_DATA="$PWD/.local/research-snapshot/workspace/hey-nyx-training-v3/embedding-feasibility"
```

The CUDA lock resolves for Linux x86_64 / Python 3.12; this Mac has no NVIDIA GPU.
`gpu_smoke.py` must pass on the destination before GPU training can be called
verified. It checks GPU discovery, three random-data fitting steps, INT8 export
and BUILTIN_REF execution. It does not measure wake recognition. Install a
host driver compatible with the locked CUDA 12 packages. Native Windows should
use WSL2/Linux for TensorFlow GPU.

The complete asset set is 1.19 GB compressed (about 1.11 GiB). All downloads have archive and per-file SHA256 verification. Default download
contains features/models; `--include-audio` also retrieves the synthetic source
WAVs, and `--include-history` retrieves historical experiment source. No key is
needed to continue fitting from these features. The historical filename
`jobs-private.json` contains only generated/public dataset provenance in this
export; it contains no device credentials or private room recordings.

## Continue fitting from an existing checkpoint

The last controlled experiment adds training-only 1..1279-sample phase offsets
without changing speed, duration, original sources, class/engine sampling or
the held-out splits. Its matched control and treatment are fully trained and
calibrated. Reproduce the treatment from its original warm start:

```sh
export WAKE_PLAN="$WAKE_DATA/portable-plans/head-temporal-phase-phase-plan-v1.json"
.local/4090-env/bin/python research/wakeword_p4/embedding/train_head.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_PLAN" \
  --initial-weights "$WAKE_DATA/head-temporal32-dropout-v1/step-20000/head.keras" \
  --output "$WAKE_DATA/new-4090-phase-reproduction"
.local/4090-env/bin/python research/wakeword_p4/embedding/evaluate_head.py \
  --data-root "$WAKE_DATA" --plan "$WAKE_PLAN" \
  --training "$WAKE_DATA/new-4090-phase-reproduction"
```

Use a fresh output directory for every experiment. These scripts warm-start
model weights; they do **not** restore the original optimizer or RNG position.
For a new recipe, copy a portable plan, record changes before fitting, bind the
chosen initial `.keras` SHA256 in `initial_weights_sha256`, then pass that exact
file with `--initial-weights`. The trainer verifies it. Preserve normal audio
speed and use the same fixed validation operating-point policy. GPU and CPU
training need not produce identical new weights; report the environment.

The source feature indexes from two older directories were already relative.
Newer extra-view indexes contained absolute Mac paths: the exporter makes only
their `path` values relative and generates plans binding those new index hashes.
`original-index.jsonl` and `portable-feature-provenance.json` retain their original
identities. **Array bytes, source labels/splits, model weights and recipes do not
change.** A complete equal-mean calibration replay from the extracted directory reproduces
the original calibration JSON exactly. Original calibration and checkpoint files remain unchanged and refer
to their original audit identities. Use the portable plans for new fitting;
do not rewrite old calibrations to pretend that their metadata is unchanged.
Original absolute `jobs[].source` paths are corpus/audit metadata for fitting,
not live dependencies. Raw-feature regeneration needs paths rebased explicitly
to the downloaded workspace or public datasets retrieved from their manifests.

## Current evidence and useful starting points

| Candidate / test | Result | Interpretation |
| --- | --- | --- |
| Frozen equal mean of DNN and temporal heads, threshold .4 | Hey Nyx 46/50 paired variants; Hi ESP 48/50 | Existing development audio, not fresh human acceptance |
| Same candidate ordinary negatives | 1/100; Hi ESP 0/100 | Remaining false wake gap |
| Same candidate English continuous | 1 event / 5.549 h; Hi ESP 0 | Development regression |
| Same candidate Mandarin continuous | 0 / 1.294 h; Hi ESP 0 | Limited exposure, not a zero-FAR guarantee |
| Latest original-app Hi ESP acoustic control | Wake, immediate request, reply and quiet pass | Physical Mac-speaker control passed |
| Latest fixed Hey Nyx full-app acoustic repeats | 3/5 wakes; all 3 complete request/reply/quiet and rearm | Two misses remain; no replacement acceptance |
| Five-minute whole-app resource test | Mean 51.3 ms / 80 ms step; no new RX overflows; byte balance -640..0 | Passed these capture/resource gates only |
| Phase continuation control vs treatment | Validation macro recall 94.05% vs 94.29% | Small validation difference; no new regression/physical result |
| Phase continuation equal mean | Validation macro recall 94.73%, threshold .3 | Original frozen mean was 95.26%; no demonstrated improvement |

Detailed reports: [`../wakeword_p4/results/`](../wakeword_p4/results/), especially
`embedding-full-app-resources-v3.json`, `embedding-head-equal-mean-v3.json` and
`embedding-phase-continuation-v3.json`. Older failures remain in Git history.

### Subsequent acoustic diagnosis

An opt-in whole-app input trace now verifies the exact PCM consumed by the
detector, frontend bytes, embedding, both heads and event policy. In one home
development-panel run, all **1,682 steps match the host byte for byte**, with
5,046 complete fragments and no reported drops. Forty seconds of steady-state
resource observation passes: mean inference51.8ms/80ms, no new hardware RX
overflows, byte balance -640..0. Private room audio stays off GitHub.

At the same recorded Mac output level100, the Hi ESP control completes its
initial directive, one reply and eight seconds of quiet. Hey Nyx misses its
paired synthetic cue: peak .2181 at the frozen .4 threshold. This is one trial,
not a new recall estimate. The original Hi ESP app/service is restored. See
`embedding-input-trace-hardware-v3.json` and `embedding-home-acoustic-level-v3.json`.
The acoustic harness now rejects muted/zero-volume playback and records output
level before/after; earlier muted playback cannot establish a detector miss.

The new `prepare_level_views.py` experiment pairs low-level training views with
the completed environmental views. Existing augmentation targets RMS .04.. .20
of PCM full scale; the new prespecified range is .002.. .02. This is a hypothesis
about acoustic robustness, not a proven cause. Original room, background, SNR
and RNG draws must match the completed reference index. The input PCM speed,
source/held-out partitions and existing views remain unchanged. Plans under
`embedding/plans/level-*.json` freeze extraction and the fitting comparison.
The initial4090 snapshot does **not** include these newly generated features;
use its existing assets until a later asset manifest explicitly includes them.
New extraction requires the original source paths to be rebased locally as
described above. No new classifier or production wake replacement is accepted.

The 4090 bundle includes all completed `head-*` model checkpoints (DNN,
temporal, mining, Qwen, environmental, pooling and phase variants), original
MicroWakeWord model/weight artifacts from training v1/v2/v3, 61,064 indexed
embedding histories across eight completed feature views, synthetic source
WAVs, frontend coefficients, per-checkpoint numerical fixtures and validation score
arrays for exact equal-mean recalibration, plus the three
hash-bound board runtime bundles. See `assets.json` for exact sizes and hashes.
The selected runtime bundle is `runtime-equal-mean-bundle-v3`; its calibration
is `head-equal-mean-validation-v1/calibration.json`.

Do not conflate correlated renderings with independent speakers. The six TTS
engines are Piper, Kokoro, Qwen, OpenAI, macOS say and Edge TTS. Reserved final
English 6.455 h and Mandarin 3.126 h remain unscored and are excluded from this
training export. Public raw negative corpora can be downloaded again using
the source manifests/scripts; their ready-to-fit embeddings are included.
Build caches, installed dependencies, redundant old derived features, NVS,
credentials, device snapshots, private microphone captures and provisioned
firmware backups stay local.

## Board and numerical constraints

Read [`../panel_wake_runtime/README.md`](../panel_wake_runtime/README.md) before
building/flashing. Runtime code is opt-in through CMake and does not switch the
production default. Preserve the single I2S producer, existing AEC/AGC call
path, ordered preroll/FIFO, 16 kHz capture and 24 kbps / 20 ms Opus packets.
The experimental idle detector alone uses raw mic samples, because a matched
three-stage capture localized the old missed cue to communication AEC.

The final embedding runtime uses a bounded asynchronous consumer, measured IRAM
placement and exact padding/LUT rewrites. It destroys/joins the idle classifier
before starting a call to recover internal task-stack space, then recreates it
when rearmed. This fixes the observed call allocation failure; it does not fix
the two missed physical wakes. Shared optimized FC dispatch/rounding mismatches
are handled by the local reference-compatible evaluator. Do not substitute
XNNPACK goldens: host extraction and numeric comparisons use BUILTIN_REF.

Only the USB development panel participated. Do not copy this machine's board
addresses, USB port, account/session, keys or NVS to another environment. Select
and verify the destination development board and save its own restore image.
The runners flash only a verified active app slot and restore it in `finally`;
never write the standalone probe's partition table over a provisioned panel.

## Experiment source archive

`historical-experiment-source.tar.gz` preserves the original per-run Python and
P4 source from the output workspaces, including the earlier backbone
reconstruction/quantization, numerical probes and acoustic harnesses. These are
historical source snapshots, with original machine-specific paths and launch
dependencies; use maintained helpers under `research/` for new runs. Private
capture headers and all room recordings/device-state outputs are excluded.
The archived upstream source retains its own notices. Provenance and licensing
references are recorded in the training and embedding research READMEs.

To export a future snapshot, run `export_assets.py --help`; every output must
use a fresh folder. Publish research assets as a prerelease, never as the latest
production firmware/OTA release.
