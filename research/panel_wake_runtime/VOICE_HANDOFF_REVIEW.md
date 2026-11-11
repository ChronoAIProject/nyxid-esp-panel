# Wake-to-call ownership review

Reviewed against `8bde4c6`, while GPU v39 training has no quality conclusion.
This is source review of the selected paths below, not an acoustic or concurrent
device execution. No firmware or audio-pipeline change follows this review.

| Requirement | Observed implementation |
| --- | --- |
| One I2S producer during a wake call | `panel_voice.c::wake_task` keeps `product_i2s_read`; `mic_task` takes its `wake_initiated` branch and consumes the processed capture FIFO without another I2S read. Explicit/remote calls instead claim audio ownership before their ordinary microphone task starts. |
| Preserve AEC/AGC and original sample speed | The wake task processes each hardware M/R capture through one AEC and160-sample AGC blocks before appending preroll/call samples. Experimental idle detection separately reads the same capture's raw mic. Call and preroll stay processed; no sample resampling is introduced. Opus remains16kHz/24kbps/20ms. |
| Preroll followed by live capture without duplicate append | `start_call(..., true)` copies the circular two-second tail in order under `wake_lock`, clears that tail, and publishes `wake_call`. The producer thereafter appends only to that call's FIFO. The consumer pops320-sample frames under the same lock. Bursts allow queued frames to catch up without changing PCM or their20ms timestamps. |
| Initial connection and reconnection have distinct buffering | `panel_voice_pcm_gate` retains pre-first-connection samples; readiness permanently marks `capture_started` for that call. Later readiness gaps/mute clear pending audio and reject gap samples. Both producer and consumer call the gate while holding `wake_lock`. |
| Idle inference does not compete with a wake call | The experimental classifier is destroyed before `start_call`; the producer, AEC/AGC and preroll remain alive. `wake_call` selects the call branch thereafter. Cleanup keeps `busy` asserted while detaching the call and setting a rearm delay, so the null classifier is not used as an idle detector in that interval. |
| Call capture lifetime is serialized | `wake_call` and capture access are under `wake_lock`; teardown stops and joins call tasks, then clears the pointer and FIFO under that lock before freeing call memory. Explicit-call audio ownership is released only when owned. |
| Diagnostic PCM cannot open a call | `panel_wake_runtime.cpp` runs its optional self-test on a separate detector, deletes it, then creates the live detector. `probe_wake_clock` records numeric results without entering call startup or preroll. |

Source references:

- [Wake producer, call consumer and teardown](../../components/panel_services/panel_voice.c):
  `wake_task`, `mic_task`, `start_call`, `call_task`, `wake_available`, `wake_yield`.
- [Audio ownership](../../components/product_support/src/product_support.c):
  `product_audio_claim`, `product_audio_claim_background`, `product_audio_release`.
- [FIFO/gate helpers](../../components/panel_services/panel_voice_audio.c).
- [Experimental detector lifecycle](panel_wake_runtime.cpp).
- [Existing FIFO tests](../../tests/host/test_panel_voice_audio.c): ordered stream,
  overflow refusal, circular preroll, readiness gaps and mute behavior.

No definite second-reader, duplicated-preroll or repeated-AEC defect was found in
these reviewed paths. This does not rule out timing, driver, transport, or
acoustic failures. Existing host tests exercise FIFO helpers, not the complete
FreeRTOS tasks or actual codecs. A successful idle120-second/model-fixture test
also does not establish active-call stability.

Before an eligible candidate is accepted, the original target still requires
matched real speech/acoustics, wake-to-call start and final-word latency,
long two-way speech, echo/reconnect behavior, overflow counters and resource
measurements against Hi ESP. Those remain unverified for v39. The dining-room
production panel is excluded; current automatic work uses no speaker playback,
microphone recording or board access.
