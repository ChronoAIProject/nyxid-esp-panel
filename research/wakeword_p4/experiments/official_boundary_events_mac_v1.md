# What the saved official events do and do not show

Mac independently checked all 26 original official points, 936 positive clip
evaluations and 267 positive events against the pinned execution input contracts
and GPU boundary description. Integer sample clocks, all signed boundary offsets,
all categories and original summary counts agreed. No model/audio execution,
threshold selection or change to the original scoring occurred.

| Original point | Any event after source starts /36 | Event after generated prefix ends /36 | Original background events /10.48h | Original FAR upper95 /h |
|---|---:|---:|---:|---:|
| Willow DEFAULT | 4 | 4 | 0 | 0.285745 |
| Willow SET 0.644 | 5 | 5 | 0 | 0.285745 |
| Willow SET 0.5 | 30 | 23 | 50 | 6.036571 |
| Hi ESP DEFAULT | 36 | 14 | 1 | 0.452489 |
| Hi ESP SET 0.644 | 36 | 17 | 0 | 0.285745 |

SET requests are labels, not assumed identical to actual getters; exact getter
float bits are preserved in the JSON. Background event counts and confidence
bounds are copied from the original producer and not independently recomputed
from raw per-call UART here. Background sample exposure was checked separately.

Hi ESP's DEFAULT 14/36 after-generated-prefix metric is not evidence of only
14 triggered clips: all 36 have an event after source onset, including 22 whose
only event precedes the generated prefix end. This explains why decreasing a
threshold can increase observed triggers while decreasing after-prefix counts.
Saved events do not prove whether refractory suppressed a later hypothetical
event. No refractory interval or causal counterfactual is inferred.

Generated prefix ends are internal VITS allocations, not measured phonetic
boundaries. An early event also does not prove correct complete-word recognition.
Both facts matter: neither the after-prefix count nor any-after-source count
alone establishes mature recognition quality. The original criterion/results
remain intact. Future recognition/latency protocols must define detection windows
and word-boundary evidence before seeing fresh scores; incomplete phrases and
confusable negatives must remain part of their qualification.

Willow's 30/36 observed events at SET0.5 also accompany 50 background events;
that point is not a same-FAR alternative to DEFAULT. None of this establishes a
qualified operating point or Hey Nyx parity. Current raw-FP32 synthetic numerical
success must first extend to the separately locked fullstream stage. Original
bounded-float recognition results remain a different output-head construction.
