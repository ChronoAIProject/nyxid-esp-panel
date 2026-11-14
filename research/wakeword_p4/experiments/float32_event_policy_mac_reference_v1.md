# Independent float event reference for the reused-development diagnostic

`float32_event_policy_mac_reference_v1.py` is opt-in research code. It changes no
historical evaluator and does not start collection, score data, choose a working
point, change a model or authorize acceptance. GPU's actual diagnostic must bind
its chosen arithmetic and all metric/collector code before consuming scores.

The event state machine matches the original pinned UINT8 policy: first complete
five-step window, 67-step cooldown, half-threshold low rearm after cooldown, and
one-based step clocks. All original scores remain present, including startup.
The only representation adaptation is to use original float32 probabilities,
convert exactly to binary64, compute `math.fsum(last5)/5`, and compare with the
fixed binary64 threshold. No byte rounding, sigmoid clipping, implicit dtype
conversion, moving binary32 sum or rolling-subtraction accumulation is applied.

For original bytes dequantized exactly as `float32(byte)/256`, this gives exactly
the original `sum(five_bytes)/1280` arithmetic. The constructed review checks
all1276 attainable five-byte sums and1380 whole-sequence/registered-grid
comparisons with the actual pinned legacy module. Separate checks cover
non-byte float precision, true probability1, plateau/cooldown/rearm, terminal
clock clamping and11 invalid representations/values. No real probabilities,
audio, development or acceptance are read by this review.

```sh
python3 research/wakeword_p4/experiments/review_float32_event_policy_mac_v1.py \
  --repository-root /path/to/repo --report /new/path/review.json
```

The result does not establish recognition quality. The official WakeNet event
engine is separate; its numeric thresholds and actual getter values cannot be
equated with custom thresholds. Compare the complete registered curves against
the same original exposure and FAR upper95 requirement, keeping all points and
the original common/cross-target negatives. A reused-development diagnostic must
not silently call a working-point selector or relabel old media as new data.

Report both any-after-source and original after-generated-prefix results,
including every original event and signed boundary offset. Neither alone proves
correct complete-word detection or measured latency: generated prefix ends are
synthetic allocation boundaries. Preserve the original criterion and keep fresh
development/independent acceptance behind their own prior protocols. Only that
later evidence plus P4 resource/first-utterance/long-call validation can establish
the active goal's mature-baseline parity.
