# C161 P4 synthetic witness validation handoff

Use the already-running C161 development-board configuration reported in Issue18;
this handoff requests evidence collection, not production deployment or a model
replacement. No human speech is needed. The metadata manifest
4090-c161-p4-consumed-witness-metadata-v1.json identifies all12 existing consumed
Mac Daniel/Samantha Nyx/Willow/HiESP bare/command witnesses, original Mac AIFF
hashes, transported WAV/PCM identities and historical detector events.
No audio, firmware or new model is included. C161 stays60952bytes/SHA
c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81.

These are descriptive verification steps, not invented numeric acceptance gates:

1. Record actual board/revision, firmware commit and model hash, runtime profile,
   frame clock, frontend and model engine. Preserve known production/restoration
   conditions. Existing boot/frame-count evidence alone is not recognition.
2. For numerical injection, Mac generates its PCM self-test from the original
   local sources and verifies PCM hashes before comparison. Use the fixed C161
   profile:16kHz, threshold235/256, smoothing5, refractory67, rearm0.5;
   continuous81600 zero PCM through frontend/model/decoder, suppress notifications
   only during prime. Do not reset at source entry. Preserve prime events, raw
   output codes, block timestamps and final-flush policy. An event disagreement
   needs the first divergent raw value/source sample; do not change the threshold.
   If an original source is unavailable, report it; regenerated speech gets a new
   identity and cannot inherit historical numerical expectations.
3. For actual speaker-to-microphone recognition, play synthetic target/command
   audio and normal-word controls on the development setup. Record source hash,
   speaker/mic, fixed distance/orientation/playback volume, environmental sound,
   complete attempts, detector events and captured input where available. An
   acoustic recording is a new waveform; do not expect equality with injected
   PCM events or label an injected-only test as microphone recognition.
4. For connected commands record the actual wake event timestamp, command-buffer
   start/end and returned audio/sample counts or downstream command result. Keep
   complete failures. Generator text is not a measured phoneme boundary; an
   event falling in a clip does not prove the first command word was retained.
5. Record loaded-model arena, free/minimum heap and PSRAM/internal memory with
   normal display/network/audio services active; model-call latency distribution,
   audio deadline/overrun counters, run duration, resets and connection status.
   Host tensor sums and short idle frame tests are not whole-system measurements.
6. If official Hey Willow is available, pin its model/runtime/default threshold
   getter and repeat each detector's own-target and common controls under the
   same acoustic setup. Retain each native event rule and clock (Willow512 versus
   C161480-sample calls); do not tune either operating point to these witnesses.
   Hi ESP remains an engineering/reciprocal control. Historical Willow4/4 versus
   C1611/4 is a small synthetic positive-control gap, not a P4 or user benchmark.

Return a metadata/report link in Issue18 with each category explicitly marked
measured, unavailable or not run. Include per-attempt results, identities,
resource logs, limitations and any blocker. Report actual daily background
sources/durations/events once; do not multiply reused recordings into new
exposure. Rare constructed nearwords remain diagnostics. No result here licenses
training on these already-consumed witnesses or claims real-user generalization.
