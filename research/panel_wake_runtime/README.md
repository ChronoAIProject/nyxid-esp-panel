# Full-application Hey Nyx experiment

This adapter runs a separately trained streaming INT8 model in the existing
panel wake task. It is an opt-in research build, not a replacement for the
production Hi ESP detector. Model files and private acoustic recordings stay
outside git.

The application retains its single 16 kHz I2S producer, communication AEC/AGC,
ordered wake lookback/FIFO, 24 kbps Opus, 20 ms packets and normal speech speed.
While idle, the experimental detector reads raw mic samples from that producer:
its MicroFrontend supplies noise reduction and PCAN. The communication AEC can
suppress its wake cues even with a silent reference. A matched development-panel
capture produces a wake at score 0.996 on raw mic, and zero after AEC/AGC; this
is one diagnostic case, not a recall estimate. The processed preroll and call
audio continue through the existing AEC/AGC path. The default Hi ESP detector
keeps its original input. No second microphone reader or recording task is added.
Two matched Marin acoustic wake/request/reply/quiet/hangup/rearm rounds pass
after this input change. The diagnostic model's offline recall/FAR still fails
the replacement target; this is integration evidence only.

```text
one I2S mic ─┬─ AEC ─ AGC ─ ordered lookback / call FIFO ─ Opus
            └─ MicroFrontend ─ Hey Nyx detector (idle only)
```

The adapter owns only feature extraction and inference. It reports the configured phrase
through the existing wake status and panel label. It does not change service
permissions, Bot selection or provider credentials.

## Build

Use the release's ESP-IDF 5.5.4 environment and dependency versions. Set the
following paths to a private model directory and its quantized validation
calibration. The directory must also contain the matching `training.json`.

```sh
idf.py -B build-hey-nyx \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_MODEL="$WAKE_MODEL" \
  -DNYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION="$WAKE_CALIBRATION" build
```

For an already provisioned development profile, include the documented
`NYX_PANEL_NODE_DEV_MODE`, `NYX_PANEL_AGENT_DEV_API_URL` and
`NYX_PANEL_AGENT_DEV_LOGIN_URL` build options. Keep all registry versions pinned
to the normal release and add only the adapter's declared dependencies. Do not
update unrelated camera, audio or UI packages while measuring a detector.

Configuration verifies the model, training report and calibration hashes,
supported topology, input context, frozen threshold and score-average window.
Reviewed calibration windows are three, five or seven model steps; older
calibrations without the field retain five steps. Invalid explicit metadata
fails configuration. Offline comparisons and the firmware must use the same
bound window. Changing it requires a new validation calibration, not tuning
against the acoustic test. The generated
`manifest.json` under `build-hey-nyx/esp-idf/panel_wake_runtime/generated`
records those identities and detector settings. A failed production-quality
calibration is still usable for an explicitly diagnostic research build; a
successful build does not imply recognition acceptance.

For an inference diagnostic, the training checkout's
`research/wakeword_p4/training/prepare_runtime_selftest.py` creates a private
model-bound binary from an existing 16 kHz mono WAV and host INT8 outputs.
It keeps at most the first four seconds, without changing sample speed. Pass
the resulting path as `-DNYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST=...` to include
the optional startup check. Configuration rejects another model's fixture,
truncated data and invalid sample/output counts. The complete firmware then
compares every output byte through its own frontend and inference path. The
diagnostic detector is destroyed before creating the live microphone detector;
test audio never enters a call. A mismatch leaves wake unavailable and is
reported in UART. Omit this option for ordinary acoustic/resource measurements.

## Temporary board test

Use a USB-connected development board with a known MAC, current app ELF hash,
active OTA slot and a verified local recovery binary. Never target the dining
room production board. Keep its existing partition table, bootloader, OTA
metadata, credentials and filesystem. Stop all other UART readers first.

1. Verify the original app is idle, connected and listening for Hi ESP.
2. Run `acoustic_voice_smoke.py --help` with the original app hash and a known
   good Hi ESP + immediate request WAV. It plays through the Mac speaker and
   checks the complete request, one reply, no extra turns during eight seconds
   of quiet, and cleanup of its own call.
3. Obtain the experimental app ELF hash using
   `python -m esptool image_info --version 2 build-hey-nyx/nyxid-esp-panel.bin`.
4. Run `run_full_app.py --help`. Supply both app paths/hashes, the explicitly
   verified MAC/slot/offset/size and a fresh private output directory. After
   `--`, supply the acoustic command using the experimental hash and Hey Nyx
   fixture. The runner checks the MAC, flashes only that app slot, verifies
   the same service selection and Hey Nyx listening, then executes acceptance.
5. The runner records UART during experimental acceptance, closes its reader
   before reflashing, restores the original app on success or ordinary failure,
   and verifies its identity, connection and original wake listener. Keep the
   runner alive through restoration; SIGTERM requests cleanup, while SIGKILL or
   removing power prevents that guarantee. Inspect `report.json` for
   `restored: true`, including after an unsuccessful acceptance command.

Run the outer runner with an ESP-IDF Python that has `esptool` and `pyserial`.
The acoustic command can use a separate environment with `pymongo`. The local
owner adapter reads test state/history and hangs up the test's own call; the
board continues using its own stored Voice key. No provider/Mac key is copied
to firmware. Keep UART, account identifiers and conversation/audio artifacts
private; publish sanitized aggregate findings only.

Compare both engines under fixed speaker placement and playback volume. A new
TTS phrase that fails the Hi ESP control cannot validate the physical setup;
retain and report such failed controls. One successful wake/request/reply is
integration evidence, not a recall estimate. Recognition acceptance additionally
needs matched voices/conditions, continuous negative exposure, human positives,
distance/noise variation, repeated rearming, immediate speech, playback echo,
reconnect/long calls and full-app CPU/RAM/flash measurements.
`acoustic_voice_smoke.py` records Mac output volume/mute state before and after
each trial and rejects muted/zero-volume playback before playing. It does not
change the Mac volume. Keep the same output device and record any externally
changed level; checking the WAV file alone does not verify physical playback.

### Frozen multi-voice acoustic measurement

`acoustic_wake_probe.py` measures wake events and rearming without recording
microphone audio or reading conversation history. Its frozen plan binds the
probe/helper hashes, exact 16 kHz PCM16 WAVs, app identities, default playback
device/volume, source onset annotations, trial order and observation windows.
It validates the explicit development service/app at each status poll, stops
only its own local wake call, and refuses unrelated active calls. A zero exit
code means the measurement completed, including misses; it is not recognition
acceptance. `report.json` retains each missed, early or onset-ambiguous event.

The completed five-voice protocol is
`plans/acoustic-matched-pilot-v1.json`. Fetch its synthetic WAVs with
`research/handoff/fetch_assets.py --asset acoustic-matched-pilot-v1.tar.gz`.
The source root is the extracted `workspace/hey-nyx-training-v3` directory.
On this protocol the original Hi ESP before/after arms trigger 1/5 and 2/5
positives; the frozen Hey Nyx app triggers 4/5. Each arm triggers 0/5 ordinary
negatives. All trials rearm. This small reused synthetic whole-app pilot,
including failed Hi ESP controls and different detector preprocessing paths,
does not establish Hi ESP equivalence. See the sanitized result linked from
the [handoff](../handoff/README.md#matched-physical-five-voice-pilot).

The frozen expanded plan `plans/acoustic-matched-extended-v1.json` measures
25 positive / 50 shared ordinary-negative trials per arm. Its completed
Hi ESP / Hey Nyx / Hi ESP bracket has 6/25, 12/25 and 5/25 positive triggers,
respectively, and 0/50 ordinary-negative triggers in each arm. One of the
12 Hey Nyx observation intervals straddles source onset, leaving 11 clearly
after onset. All 225 trials rearm and the original app/service/listener are
verified afterward. The low, variable Hi ESP controls and different detector
preprocessing prevent a parity claim. The synthetic source asset is
`acoustic-matched-extended-v1.tar.gz`; the
[expanded report](../wakeword_p4/results/embedding-acoustic-matched-extended-v3.json)
retains the ambiguous event and every failed control. No threshold or
playback setting was chosen from either physical protocol.

For a new environment, freeze a new plan before measuring: explicitly select
and verify its development device/app, playback device, unchanged source PCM,
placement and arm order. Do not edit the completed plan or tune thresholds from
its results. The current playback adapter uses macOS `afplay`/system tools.

### Matched Hi ESP input diagnostic

`NYX_PANEL_RESEARCH_HIESP_INPUT=processed` and `=raw` build two isolated
Hi ESP controls. Both require `NYX_PANEL_NODE_DEV_MODE=ON`, reject an
experimental Hey Nyx model, and pin the measured dependency versions. Empty
is the normal build. `processed` retains the normal idle AEC/AGC input;
`raw` feeds the detector the same single producer's raw microphone samples.
Both keep processed preroll, the active-call AEC/AGC and the ordered FIFO.
They use the same embedded Hi ESP weights, DET_MODE_95 and model lifecycle.

Build both with the same sdkconfig and
`-DCMAKE_PROJECT_nyxid-esp-panel_INCLUDE=.../embedding_size_profile.cmake` when
using the existing private size profile. The profile retains the calculator's
release optimization even when no experimental runtime target exists. Compare
the dependency locks, generated sdkconfig headers and compile-command arrays:
the only command difference must be `NYX_PANEL_RESEARCH_RAW_HIESP=1` on
`panel_services`. Save each app image before changing the cache input mode.

The frozen `plans/hiesp-input-{processed,raw}-v1.json` plans reuse the first
three positive and first two ordinary-negative source indices for each of
five existing synthetic voices (15 positive,10 negative trials per arm).
They retain the parent's randomized order and source bytes. Run processed,
raw,processed through `run_full_app.py --experimental-wake-word 'Hi ESP'`,
with each plan passed to the unchanged `acoustic_wake_probe.py --arm hi_esp`.
The runner restores and verifies the original app/service/listener after
each arm. Rebuilding changes the app identity: freeze a new plan rather than
loosening identity checks or editing a completed measurement.

This controls the idle detector input, not the trained wake phrase or model.
It can diagnose whether different input processing confounds the comparison;
it cannot establish Hey Nyx equivalence or continuous false events/hour.

The completed bracket yields 2/15,8/15,2/15 positive triggers, and 0/10
ordinary-negative triggers per arm. The processed repeats agree on every
trial. Raw gains seven positive clips and loses one; all 75 trials rearm.
All three swaps restore and verify the original app, selected service and
Hi ESP listener. The [sanitized report](../wakeword_p4/results/hiesp-input-acoustic-v1.json)
retains each voice's hits/misses and every observed event interval. Its
aggregation verifies frozen plan/trial order, source hashes, playback
conditions, app identities, restoration, the raw-input UART marker and
absence of UART panic/read errors before publishing counts. No room audio,
account identifiers or private UART is published.

This isolates an important confound in the earlier physical Hey Nyx bracket.
It also retains Hi ESP's remaining raw-input misses; neither result can be
promoted to human acceptance. The next physical model comparison must use
the same raw idle source for both engines and keep the unchanged processed
communication path. Independent continuous-negative and human/room evidence
remain separate requirements.

### Raw-input engine comparison

`plans/raw-engine-comparison-v1.json` freezes a new Hi ESP / Hey Nyx / Hi ESP
bracket after the input-control result. Both idle detectors receive the
single producer's raw mic; communication audio keeps its original AEC/AGC
and ordered preroll/FIFO. The Hi ESP arm uses the pinned raw-input control;
the Hey Nyx arm uses the already measured equal-mean runtime. Neither model,
threshold nor source waveform changes during the bracket.

Each arm has 35 trials in the same logical randomized order: three positive
forms per each of five reused voices, two shared ordinary sentences, and the
shared near-word fixtures `Hey Nick` and `Hi ESB` per voice. Categories remain
explicit in each trial so ordinary and near-word false counts cannot be
pooled away. These are reused generator-labelled synthetic regression clips,
not human pronunciation truth or new independent speakers.

Run the unchanged `acoustic_wake_probe.py` with `--arm hi_esp` or `hey_nyx`
through `run_full_app.py`, binding each plan's exact app identity and wake word.
Before the Hey Nyx acoustic command, run `resource_smoke.py --seconds 40
--require-capture-counters --capture-backlog-bytes 27264`, using that runner's
private UART file. This checks current idle cadence and hardware byte balance
without playing or recording room audio. Do not infer call/reconnect or
recognition acceptance from that idle check. Restore and verify the original
app/service/listener after every arm, including any failed resource check.

```sh
python research/panel_wake_runtime/acoustic_wake_probe.py \
  --plan "$WAKE_ACOUSTIC_PLAN" --arm hi_esp \
  --source-root "$WAKE_SOURCE_ROOT" \
  --status-url "$DEV_PANEL_STATUS_URL" \
  --expected-app-sha256 "$ORIGINAL_APP_ELF_SHA256" \
  --expected-slug "$DEV_SERVICE_SLUG" \
  --admin-module "$LOCAL_OWNER_ADAPTER" --admin-base "$LOCAL_API_BASE" \
  --output "$NEW_PRIVATE_OUTPUT"
```

Run Hi ESP before, then the `hey_nyx` arm through `run_full_app.py` with its
experimental ELF hash, then Hi ESP after the runner verifies restoration.
Keep the runner alive until its `restored: true` report. Stop-on-wake measures
neither complete conversation nor independent continuous false-event exposure;
use `acoustic_voice_smoke.py` for first-utterance/reply/quiet acceptance.

### Private input and numerical tracing

For an embedding-runtime diagnostic, set
`-DNYX_PANEL_EXPERIMENTAL_WAKE_TRACE_IP=<receiver-private-LAN-IP>` and optionally
`-DNYX_PANEL_EXPERIMENTAL_WAKE_TRACE_PORT=30282`. This requires the isolated
`NYX_PANEL_NODE_DEV_MODE=ON` profile. The default empty address compiles out the
trace. Never enable it in a production or OTA release.

Start `trace_capture.py --source-ip <development-panel-IP> --output
<fresh-private-directory> --seconds 300` before booting the diagnostic app.
The bounded receiver accepts only that source IP. The detector copies its
already consumed PCM, quantized frontend, embedding, classifier bytes and
event state; it does not open the microphone or advance an extra model.
Three nonblocking UDP fragments per 80 ms step each fit a standard IPv4 MTU.
There is no retransmission. Keep every capture private: it contains room audio.

After the runner has restored the original app, stop the receiver and run
`replay_trace.py --help` in the NumPy/LiteRT research environment. Supply the
captured `packets.bin`, original unfused reference backbone, exact runtime
bundle, FFT window and mel-filter coefficients. The replay rejects incomplete,
duplicate, conflicting or corrupt records, then compares each numerical stage
and the frozen event policy using BUILTIN_REF. A host-generated trace only
tests the harness; it is not evidence of device numerical parity. Record trace
packet drops and scheduling/stack overhead alongside any hardware result.

## Frozen embedding diagnostic

`NYX_PANEL_EXPERIMENTAL_EMBEDDING_BUNDLE` selects a separate research adapter
for an INT8 streaming speech embedding plus two frozen classifier heads. It
keeps the same single microphone owner and processed call/preroll path. Its
native512-point FFT/mel frontend matches the embedding training pipeline;
MicroFrontend NR/PCAN belongs to the earlier adapter, not this one.

The private bundle contains `backbone.tflite`, `dense.tflite`, `temporal.tflite`,
`native-coefficients.h` and `manifest.json`. The manifest binds every file's
size/hash, frozen calibration, exact80ms cadence and equal-mean event policy.
The dense model must also match `NYX_PANEL_EXPERIMENTAL_WAKE_MODEL`. A temporal
layout correction is allowed only with the original calibration identity and
the recorded299host/897P4 exact-output checks. `prepare_embedding_runtime.py`
rejects a changed artifact, mismatched member, missing numerical evidence or
unfrozen threshold before generating compressed model data. No model or audio
artifacts are committed here.

Each model expands once into PSRAM. The current comparison reserves96KiB
internally for backbone tensors,16/24KiB in PSRAM for the heads, plus a16KiB
internal variable arena. The initial all-PSRAM placement is recorded separately
and was not faster. These reservations exclude model
bytes, the detector, interpreter objects, task stacks and the rest of the app.
The first temporal convolution's spatial-width layout needs excessive ESP-NN
scratch. The exact feature-channel reshape avoids that allocation and uses
18,956tensor bytes in its standalone probe. Classifier-head and backbone
numerical/performance checks live in the training checkout's `embedding/`
tools and sanitized `results/` reports.

The first full-app performance build exceeds the existing OTA slot. A private
size experiment uses a copy of the release sdkconfig with
`CONFIG_COMPILER_OPTIMIZATION_SIZE=y` and performance optimization disabled.
Pass it explicitly with `-DSDKCONFIG=...`. Also pass
`-DCMAKE_PROJECT_nyxid-esp-panel_INCLUDE=.../embedding_size_profile.cmake` to
retain `-O2` for the scalar frontend/FC adapter and calculator. Pinned TFLM
already supplies its own `-O3`. The normal release sdkconfig is unchanged.
Always provide the model, calibration, bundle and isolated development-profile
options together when configuring a fresh build: a failed CMake configure may
discard cached values. Inspect generated compile commands and app size before
using `run_full_app.py`; never enlarge a provisioned board's partitions to make
an experimental binary fit.

For a resource-only acceptance command inside that runner, use
`resource_smoke.py --status-url ... --expected-app-sha256 ... --uart
<runner-output>/uart-private.log --seconds 30 --output <fresh-private-report>`.
It verifies continuous app identity, connection, idle listening and uptime,
records heap snapshots, and requires two complete100-step timing windows with
mean/p95 inference below the80ms audio cadence. The adapter records actual
whole-app timing, rather than summing standalone benchmarks. Passing this
check says nothing about recognition quality or long-call audio stability.

The resource gate also compares the wall-clock interval between timing windows
against the corresponding input duration (100steps =8seconds,3% tolerance).
It additionally limits aggregate drift to1%; the original per-window gate
alone admitted a persistent2.3% lag in the verbose synchronous experiment.
Fast model calls alone are insufficient if the complete microphone pipeline
falls behind. Stage timings and bounded backbone operator profiling attribute
costs without recording microphone PCM. Use the optional experimental-only
`-DNYX_PANEL_EXPERIMENTAL_WAKE_TASK_CORE=0` or`1` for controlled scheduling
comparisons; the normal release task placement remains unchanged.

The first all-PSRAM, internal-backbone and core0 whole-app comparisons all fail
at roughly118ms per80ms step. Pinning/memory placement did not resolve this.
The bundle's optional`backbone_layout` records the independently verified
PAD+VALID rewrite:114host streaming cases and32,832P4 output values must match
exactly, with unchanged constants/accumulator bounds and audio cadence. The
rewrite changes dispatch into ESP-NN's existing vector implementation; it does
not change recognition weights or the selected threshold.

`-DNYX_PANEL_EXPERIMENTAL_WAKE_ASYNC=ON` measures a sole model consumer on
core1. The existing wake task still owns all I2S/AEC/AGC, preroll and active
call capture. Priming remains synchronous. Four queued1280sample blocks are
copied in order, with sequence numbers/CRC checked before consumption; there
is no resampling, overwrite or skipped frame. A full queue fails the detector.
Destruction stops and joins the consumer before releasing its queue/models.
Queue control/atomics and the additional16KiB task stack are internal; only
bounded sample storage uses PSRAM. This experiment initially keeps8seconds
of audio in8seconds of wall time, but overflowed after about54seconds in the
first run. It remains rejected, rather than increasing capacity to hide drift.

`-DNYX_PANEL_EXPERIMENTAL_WAKE_IRAM=ON` separately maps the hot adapter and
pinned convolution objects into internal memory using`embedding_iram.lf`.
This tests code-fetch/cache effects without editing dependency sources. It
does not make the detector safe while caches are disabled: model data may
still be in PSRAM. Measure heap and use at least120seconds for follow-ups;
the first30seconds would have missed the asynchronous queue overflow.

The GUI's250ms voice refresh originally rewrote unchanged local colors. LVGL
still invalidates these styles, causing unnecessary full-app drawing load.
Compare each exact property/selector before setting its new color. With that
fix, a controlled120s async/IRAM run falls from mean77ms/p95~115ms to
mean52ms/p95~55ms. A separate synchronous/noIRAM run has mean50ms, but verbose
operator/task UART logs stall capture by2.3% in aggregate. Turning those
logs off brings the120s aggregate drift below1%. Keep
`NYX_PANEL_EXPERIMENTAL_WAKE_PROFILE=OFF` (default) for acceptance; enable it
only to attribute bottlenecks. Recognition and hardware-loss proof remain
separate from those timing measurements.

`NYX_PANEL_EXPERIMENTAL_WAKE_CAPTURE_COUNTERS=ON` adds optional ISR observers
to the existing I2S RX channel at creation, using a research-only linker wrap.
They count received bytes and driver queue overflows without inspecting PCM,
reading the channel, changing DMA settings or opening another audio owner.
Use `resource_smoke.py --require-capture-counters` to reject missing counters
or any new overflow between complete timing windows. Also supply the verified
`--capture-backlog-bytes`: for the measured16bit stereo/256sample producer,
6x240frame DMA capacity is5760bytes, the current block is1024bytes, and the
optional4x1280sample model FIFO adds20480bytes. Thus the synchronous bound is
6784bytes and the asynchronous bound27264bytes. The gate compares each prefix's
received bytes with the detector's exact input consumption, not just consecutive
window deltas. IDF's nearly-full RX read path can discard a partial DMA tail
without reporting queue overflow: the five-minute synchronous run accumulated
131520unaccounted bytes (2055ms), despite zero new overflow callbacks. It is
rejected; its earlier timing-only pass is not proof of lossless capture.
Startup and synthetic
model priming happen before that measured steady-state interval and may
already have accumulated unconsumed RX buffers; report the initial counter
separately. A timing pass alone does not demonstrate lossless capture.

The subsequent five-minute async/IRAM run with the conditional-style fix passes
all idle gates: mean51.894ms, final p95 54.716ms, queue highwater1, and
288000ms of input consumed in287999ms of measured wall time. Hardware RX
overflow stays at its startup value826; the prefix byte-balance bound passes.
These results establish measured steady-state capture/scheduling only.

The physical Marin Hey Nyx fixture then wakes the panel, but call setup fails.
A matched diagnostic build adds allocation error reporting and confirms that
the49152byte microphone task stack cannot fit the31744byte largest internal
free block. The idle classifier still owns its internal arena and consumer
stack. Do not shrink the Opus stack, alter PCM or increase audio queues to
work around this. The research-only fix destroys/joins the idle detector before
creating the call; AEC/AGC, the sole I2S producer, preroll and ordered call FIFO
remain alive. Recreate/prime the detector when listening rearms, reusing the
existing capture buffers. Repeated immediate-request/reply and rearm checks
must independently pass before claiming full-chain acceptance.

The release-before-call image fits the same slot at6450400bytes and passes
another300s hardware/timing check (mean51ms, prefix byte balance-640..0,
no new RX queue overflow). Its first acoustic attempt misses entirely; retain
that failure. A fresh original-app Hi ESP matched control then passes complete
immediate request/reply/8squiet. Five prespecified repeats of the unchanged
Hey Nyx fixture subsequently produce3 complete passes and2 missed wakes.
All five rearm; detected calls have no task-allocation failure and expose a
126976byte largest internal block before setup. Active/ended GUI snapshots
render normally. The original app/service/Hi ESP restores and verifies after
both failed overall experiments. This validates the memory-lifecycle fix,
not reliable replacement wake recognition; model/threshold/audio are unchanged.

Set the outer `run_full_app.py --command-timeout` longer than all of its child
checks combined; a300s resource run already exhausts the default300s limit.
For five-minute resources plus acoustic/GUI checks use an explicit900s bound.
The runner records failure type/message, gives the isolated acceptance process
group SIGTERM for its owned-call cleanup, then restores the original app.
An actual nested-child timeout fixture verified cleanup before the runner
returned; ordinary resource/acoustic results remain separate.
