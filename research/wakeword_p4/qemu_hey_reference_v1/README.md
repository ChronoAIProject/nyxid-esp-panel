# Separate classic ESP32 QEMU reference ABI v1

ESP-SR 2.6.1 / IDF 5.5.4; no board. Historical replay remains unchanged.
This project was built in `.local/wakeword/hey-reference-qemu-project-v1`
with the original downloaded managed components and external model parent
`.local/wakeword/hey-reference-external-models-v1`, containing unchanged
`wn9_hiesp` and `wn9_heywillow_tts`. Use the adjacent classic probe build
instructions and project name `nyx_hey_reference_v1`.

Input header `<IIfI`: magic `0x48595231`, original uint32 sample count,
binary32 threshold request, flags. Bit0 selects DEFAULT (no setter); bit1
selects Willow, otherwise Hi ESP. Other flags rejected. SET range .4..9999.
Getter is independently recorded as nine significant digits and binary32 bits;
finite 0<getter<1 required. DEFAULT is never equated with SET .644.

Each fresh engine receives 63 zero prime calls; every return is recorded
separately. Complete source follows without internal reset, sequential512-sample
calls, final-only zero padding. All raw positive events are emitted without a
64-event cap, timestamps clamped to original samples. Source CRC excludes
prime/padding. No one-hour source limit and no whole-file firmware allocation;
actual long-stream runtime certification is still pending.

`run_zero.py` permits only freshly constructed16001-sample exact zeros. It
checks engine identity, getter bits/decimal, setter status,63prime returns,
32input calls, CRC, no positive zero events, and stage104 after destruction.
It never opens any development/acceptance audio. Four actual zero reports/UART
and build/model/library hashes are preserved under `results/4090-hey-qemu-reference-*`.
Those runs are exploratory runtime characterization, not a frozen complete
threshold sweep, speech/FAR certification, P4 equivalence or acceptance.

A real-data entrance remains to be implemented with exact authenticated input
manifests and authorization; do not repurpose zero mode to open held-out data.
