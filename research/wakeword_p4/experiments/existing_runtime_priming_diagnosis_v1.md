# Diagnose recognition with the existing startup path

The fixed-final speaker-coverage models failed the original cold-start consumed
diagnostic (`02bd43de`). Preserve that result. Some events occurred before the
speech source began; this identifies an initialization question, not its cause.

The panel already primes its detector before advertising microphone readiness:

- `components/panel_services/panel_voice.c`, `prime_wake_model`, feeds zero PCM
  through `detect` and ignores positive detections during that stage. It retains
  the detector's frontend, inference and event state afterward.
- The default WakeNet path requests two seconds. Experimental models supply
  `panel_wake_runtime_prime_samples()`.
- `research/panel_wake_runtime/prepare_runtime.py` derives the experimental
  request as `(input_frames + 200) * 160`. The caller processes complete detector
  blocks, rounding the requested count up. The complete225 preparer uses the
  corresponding 68,000-sample request. Its profile does not admit context309.

Before adding another firmware startup mechanism or retraining, diagnose the
existing priming behavior on consumed material. Freeze the exact model and
unchanged complete inputs. For context309 and 480-sample blocks, the existing
formula requests 81,440 samples and actually feeds 81,600 samples. This follows
source geometry, with no duration selection from scores.

Feed that zero PCM and then the complete original PCM through one continuous
frontend and model. Preserve frontend adaptation, incomplete three-frame input,
rolling predictions, arming and cooldown across the transition. Concatenating
cached features or resetting at the real-input boundary changes this behavior.
Record all priming outputs and events separately; the app ignores detections
during priming. Count every event after real input begins, retain the complete
real-input exposure and report any missed first phrase. Shift original source
boundaries and event clocks by the declared priming amount explicitly.

Keep the original cold-start curves beside this new consumed diagnostic. Use
the unchanged threshold grid and preserve failures. A useful result would
justify a model-specific P4 profile and fresh development, independent acceptance
and actual hardware tests; it would not qualify the model by itself. No
CPU/GPU or floating-point score agreement is required for this question.

ESPHome commit `904f0df92034ee70233f6f225c65a5784fb2a6a8`,
`esphome/components/micro_wake_word/streaming_model.cpp`, also has startup/cooldown
handling in `reset_probabilities` and `ignore_windows_`. This is an engineering
reference, not a reason to copy its constants or add a second priming stage.

Coordination: [Issue #13, runtime evidence and proposed diagnostic](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13#issuecomment-6884555587).
