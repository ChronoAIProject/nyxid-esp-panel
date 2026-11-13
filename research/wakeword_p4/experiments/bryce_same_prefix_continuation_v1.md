# Same-prefix continuation diagnostic v1

This post-hoc experiment uses only the consumed 42 native Kristin/Norman sources,
126 renders, saved v30 traces and frozen historical policy. It is neither a new
development gate nor an acceptance test. It addresses a specific ambiguity:
separately synthesized bare and command utterances have different keyword
acoustics, so their different detection rates do not isolate continuation.

Before any intervention inference, authenticate every original and reproduce
all 126 original WAV containers, frontend arrays, UINT8 traces and event lists
exactly. Use the source resampled to 16 kHz at native speed. For every row,
including all positives and confusers, retain original plus three controls:
zero the dry source after its generator-prefix endpoint plus 0, 100 or 200 ms.
Keep original whole-source gains, group headroom, noise samples, duration,
causal RIR and surviving prefix RIR tail. Do not renormalize. A cutoff beyond
the dry source end leaves the original unchanged. Explicitly restore exact
original PCM before each cutoff to remove FFT roundoff differences, and record
any such corrections. Assert causal frontend and model score prefixes match.

Report all cutoffs and all cohorts, whole-clip and original generator-reference
window events, gains/losses of detections, and raw/five-score peaks. Keep the
original threshold 235/256, five-score mean, half-threshold rearm, 67-step
refractory, 56160-sample prime, 480-sample blocks and final padding. No model
selection, threshold search, cutoff selection or modification to firmware.

The endpoint is a generator token-duration reference, not verified acoustic
phonetics. A hard cutoff can remove real keyword tail or make an artificial
boundary; unchanged noise/RIR do not remove that limitation. Any gained
detection demonstrates sensitivity under this intervention, not a deployable
truncation policy, sole training cause, human recall or generalization.

Commit source and JSON pins before model inference. Example:

```sh
python research/wakeword_p4/experiments/bryce_same_prefix_continuation_v1.py \
  --protocol research/wakeword_p4/experiments/bryce_same_prefix_continuation_protocol_v1.json \
  --data /path/to/original/data \
  --study /path/to/bryce-v30-historical-baseline-v1 \
  --model /path/to/original/hey_nyx.tflite \
  --output /path/to/new/diagnostic-output
```

## Completed diagnostic

Inference used the source/protocol frozen at commit `7d2e014`. All 126 original
WAV containers, frontend arrays, scores and events reproduced exactly before
the three interventions were scored. All causal PCM/frontend/output prefixes
matched. The three cutoffs changed 126, 108 and 45 PCM clips respectively;
cutoffs past the dry end correctly left originals intact. Maximum causal-prefix
FFT floating difference was 4.44e-16; no quantized prefix value needed repair.

| v30, unchanged historical policy | Original | Stop at reference | +100 ms | +200 ms |
| --- | ---: | ---: | ---: | ---: |
| All target reference-window hits | 14/36 | 16/36 | 14/36 | 14/36 |
| Kristin bare | 9/9 | 9/9 | 9/9 | 9/9 |
| Kristin command | 3/9 | 5/9 | 3/9 | 3/9 |
| Norman bare | 2/9 | 2/9 | 2/9 | 2/9 |
| Norman command | 0/9 | 0/9 | 0/9 | 0/9 |
| Nyx confuser false clips | 7/54 | 7/54 | 7/54 | 7/54 |
| Hi ESP false clips | 0/36 | 0/36 | 0/36 | 0/36 |

Only two Kristin command renders gained events, both at the zero-offset cut.
The 100/200 ms controls changed no event list. Every Norman command trace
remained at UINT8 floor zero under all three interventions. Removing post-wake
speech therefore did not explain the main Norman failure under these controls.
The two gains demonstrate local right-context sensitivity; an artificial
boundary at an uncertain generator endpoint could contribute. This provides
no basis for adding silence, waiting, truncation, or threshold changes to the
device. The erased command-positive metadata finding remains a separate
training-coverage hypothesis, not a proven cause or restored coverage claim.

The companion saved-score auditor recomputes all 504 trace event lists with
integer five-score sums and independently rechecks causal score prefixes.
It does not rerun the frontend. Full scores, original reference scores, model,
protocol and pinned source are provided in the diagnostic handoff archive;
the original input dataset is referenced by its existing byte-pinned asset.
