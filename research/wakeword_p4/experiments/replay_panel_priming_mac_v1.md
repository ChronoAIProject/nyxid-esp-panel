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

## Actual final-model subset result

The Mac subsequently ran both original final context309 models against all108
already-consumed WAVs present in the packet, in both cold and primed modes.
The original138 thresholds remain unchanged. These inputs contain36
Hey Nicks target clips,36 Hey Willow clips and36 Hey Muse clips; the target
includes9 bare phrases and27 phrases followed by commands. They are synthetic
diagnostic inputs, with no continuous background or independent acceptance.

Priming removed every observed pre-source trigger at all original thresholds.
It did not solve discrimination: broad still activated on2–21/72 cross-word
clips, narrow on11–25/72, depending on threshold. Target detections ranged
4–32/36 and11–36/36 respectively. At thresholds with at least32/36 target
detections, cross-word triggers were at least20/72 and24/72. That count is a
descriptive tradeoff, not a newly selected working point or qualification rule.
No practically useful candidate is established by these results.

The [result](../results/mac-multispeaker-existing-priming-subset-v1.json)
links a lossless archive containing the original plan, every inference output
and the full curves. It contains no audio, models, credentials or device state.
Reproduce using the unchanged original model handoff and consumed packet:

```sh
"$WAKE_PYTHON" research/wakeword_p4/experiments/replay_panel_priming_subset_mac_v1.py \
  --plan "$WAKE_PLAN" --packet "$WAKE_PACKET" --models "$WAKE_MODELS" \
  --grid "$WAKE_GRID" --output "$WAKE_OUTPUT"
"$WAKE_PYTHON" research/wakeword_p4/experiments/summarize_panel_priming_subset_mac_v1.py \
  --source "$WAKE_OUTPUT/result.json" --output "$WAKE_OUTPUT/summary.json"
```

Use the archived `plan.json` and its `thresholds` list for the grid file.
`WAKE_PACKET` is the consumed packet root, with an `audio` directory;
`WAKE_MODELS` contains the two original TFLites and their handoff manifest.
The original session50845 completed with exit0. This run changes no firmware,
does not access fresh data and does not replace the GPU's full-background run.

## Original v30 model as a practical reference

While waiting for the new final model, the original v30 bytes were run through
the same108 complete consumed inputs, using their original threshold0.91796875
and current panel event policy. Its actual context150 preparation feeds56,160
zero samples. No threshold was scanned or selected. Original session6787
completed with exit0; the plan and every output are archived in the
[reference result](../results/mac-v30-existing-priming-reference-v1.json).

At that fixed point, the primed path detects25/36 target clips:8/9 standalone
and17/27 followed by commands, with0/72 cross-word triggers and no pre-source
events. All25 target hits are after the generated prefix support. By carrier,
it detects0/9 `turn on the light`,9/9 `what time is it`, and8/9 `set a timer`.
This identifies a concrete connected-command weakness in these existing
synthetic inputs; it does not establish its acoustic or training cause.

The cold path is retained as an initialization comparison. No agreement between
the two paths is required. This is a model/threshold behavior reference under
the current panel event policy, not a claim that every historical frozen
evaluator is reproduced. There is no new audio, background FAR, independent
acceptance, acoustic playback or hardware change, and this reference does not
meet the full goal merely because no cross-word clip fired.

## Final causal candidate: practical replay

The original 60,952-byte final causal INT8 model was replayed over the same
108 existing WAVs, retaining every input and the deployed 81,600-sample
zero-PCM priming. Original local session6092 exited0. No speakers, microphone,
new synthesis, flash or numerical-matching test was involved.

At the existing reference threshold0.91796875, the deployed primed path detects
32/36 targets:9/9 bare and23/27 with commands. By command it detects5/9 light,
9/9 time and9/9 timer. It also falsely wakes on6/72 cross-word clips, all
“Hey Muse, what time is it.” The original v30 reference detects25/36, including
8/9 bare and17/27 commands, with0/72 cross-word clips. Thus recall improves,
but practical discrimination still needs work. This result does not warrant
replacing the board model.

All138 original thresholds are retained in
`../results/mac-causal-final-int8-existing-priming-v1.json`; none has both zero
cross-word triggers and zero pre-source triggers. The companion archive has
the plan, per-input results and raw score traces, without audio or model files.
Cold is an initialization comparison, not an extra requirement; it is not
necessary for its scores or events to match the deployed path. There is no
background FAR, fresh acceptance or hardware-quality claim from this subset.

The user's usability criterion takes precedence over older research notes
that required FP32/INT8, CPU/GPU or gradient numerical agreement. Those
historical conditions must not be revived as candidate admission gates.
Further work should target missed intended wake words, false wakes, connected
commands, response latency and board stability.
