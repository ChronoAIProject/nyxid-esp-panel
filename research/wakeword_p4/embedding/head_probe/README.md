# Classifier-head numerical probe

This standalone development firmware tests arithmetic, placement, memory and
timing. It has no microphone, network or credential configuration. It is not a
wake recognition or whole-application acceptance test.

Prepare a private project with `../prepare_head_probe.py`, supplying the frozen
head model and exactly512 original training-only PTQ histories, plus their
expected SHA256 identities. The generated project includes256 real histories
and43 byte-boundary/random cases. Its reference outputs use LiteRT BUILTIN_REF.
Keep generated models, arrays, builds and UART captures outside git.

The project pins IDF5.5.4, esp-tflite-micro1.3.3~1 and ESP-NN1.1.2. It uses
P4 at360MHz, PSRAM at200MHz and development-panel UART GPIO37/38. Match these
settings when comparing to the cached-backbone probe. Build the app with
`idf.py build`; never flash the generated partition table or bootloader over a
panel installation.

Run `../run_head_probe.py --help` for required explicit development-board MAC,
LAN status, verified app-slot offset/size, original restore image and artifact
SHA arguments. The runner checks an idle connected original application and
USB chip identity, writes only that specified app slot and restores the
original image in `finally`. It verifies original ELF identity, active slot,
selected voice service, Nyx connection and Hi ESP listening after restoration.
Power loss or SIGKILL can prevent software restoration; retain the backup.

All three placements must complete every fixture with zero output-byte
differences. Missing/duplicate placements, partial invocation counts, invalid
resource results or a firmware failure marker reject the result. The runner
still restores the original app after numerical failure. Do not relax a gate
after inspecting outputs.

The first probe exposed a pinned upstream optimized-FC dispatch problem: common
preparation creates per-channel multiplier arrays, while the ESP-NN evaluator
unconditionally uses uninitialized per-tensor multiplier/shift. Internal versus
PSRAM placement can therefore change incorrect outputs. A local registration
shim in `reference_fc.h` retains the quantization metadata and explicitly
evaluates each channel. No shared dependency source is modified.

An additional synthetic input exposes a one-byte difference in the last
per-tensor ESP-NN FC. An optional `idf.py -DNYX_HEAD_PROBE_TRACE=ON build`
records each operator's output for fixture269. The trace matches all preceding
intermediates and localizes the difference to that FC, followed by sigmoid.
The pinned generic MCU reference FC still has this difference, so switching
the evaluator alone does not fix it. A same299-input arithmetic audit shows
that the host uses single-rounding requantization: zero different values over
19,136+19,136+299 FC outputs. Double rounding differs in27 second-FC values
and one final-FC value. The shim therefore applies single rounding locally,
handles both scale modes, computes scalar scale products in double precision
and rejects shapes/bias bounds that could overflow int32 accumulation.
Trace UART time is included in that invocation and must not be reported as
inference performance. A nontracing probe is required for the final result.

The shim repairs numerical compatibility only. It does not improve an offline
model's measured wake recall. Whole-app flash fit, scheduling, real acoustic
quality, immediate speech, rearming, long-call echo and reconnection remain
separate gates.
