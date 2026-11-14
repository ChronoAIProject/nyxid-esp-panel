# Replay the panel's existing priming on a Mac

`replay_panel_priming_mac_v1.py` compares a fresh cold detector with the existing
panel initialization: zero PCM first, then the complete original input, using a
single frontend and uninterrupted model/event state. It makes no audio output,
microphone recording, training or device changes.

Use previously consumed input and an already fixed threshold list. For example,
the frozen v30 control uses `{"thresholds": [0.91796875]}` and context150. For
another model, use its own training context and preregistered threshold grid;
this tool does not select either. With the existing research Python environment:

```sh
"$WAKE_PYTHON" research/wakeword_p4/experiments/replay_panel_priming_mac_v1.py \
  --model "$WAKE_MODEL" --wav "$WAKE_WAV" \
  --context-frames "$WAKE_CONTEXT" --threshold-grid "$WAKE_GRID" \
  --output "$WAKE_OUTPUT"
```

The output directory must be new. `inputs.json`, `progress.json` and `result.json`
retain identities, package versions, the formula-derived priming amount and all
event points. Each phase has a compressed trace of every UINT8 output and its
global/source sample clocks. Priming events remain in the report but are ignored
by the app, with smoothing, arming and cooldown retained. Real-input events and
the complete original exposure are preserved. Negative source clocks identify
priming, rather than clamping it into the first microphone sample.

The original PCM is unchanged. One continuous frontend processes zero PCM and
the original PCM together; the final hop/feature padding occurs only at the end
of that combined stream. The three-frame accumulator is never flushed at the
join. Replay elapsed time is host processing time, not measured wake latency.

The first already-consumed Hey Nicks control, frozen v30 and its existing
threshold/context were actually replayed on the Mac. Both phases retained all
58,175 real samples and detected at real sample43,040; priming processed56,160
zero samples first. The [control record](../results/mac-existing-priming-control-v1.json)
includes all output traces and the truncated-input rejection check. This is a software control for this runner, not evidence
that the new context309 models recognize correctly. Their original small models
are now available in the [GPU handoff](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/13#issuecomment-6884892185).

The shared `uint8_event_policy_v1.py` is copied unchanged from GPU commit
`02bd43de`; this adds no new event arithmetic or historical evaluator changes.
