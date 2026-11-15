# Fresh Melo development: source coverage improves recall but fails confusion

The fixed two-candidate experiment is complete. SpeechT5-final20000 wins the
preregistered joint development selection but fails the common-confusion gate.
Do not open acceptance, fall back to c161, replace the threshold or promote either
model. Frozen v30 remains unchanged. This failed experiment does not complete the
broader synthetic wakeword research goal.

| Detector / operating point | Own target /72 | Commands /54 | Common confusion /288 | Background events |
|---|---:|---:|---:|---:|
| Frozen v30, 0.91796875 | 5 | 5 | 1 | 0 |
| c161, development-selected 0.8984375 | 60 | 44 | 55 | 0 |
| SpeechT5-final20000, selected 0.3546875 | 70 | 54 | 109 | 0 |
| Unchanged official Hi ESP, DEFAULT actual 0.6299999952316284 | 33 | 18 | 0 | 0 |

All detectors received the same 432 complete synthetic presentations and 42 whole
background recordings, 7.902862222222223 hours. Each background result has the same
one-sided95% Poisson FAR upper bound 0.3790692776004911/h. These are shared
development hours, not four independent exposures or acceptance evidence. Two
config-authenticated Melo conditioning IDs and correlated speeds/conditions are
not a population estimate or proof of independent physical speakers. Training
used no Melo material. Reserved EN-INDIA/EN-AU acceptance remains ungenerated.

Targets are each detector's own phrase: Hey Nicks. or Hi E S P. Common negatives
are Hey Nick/Next/Mix/Muse. New-model errors break down as14/35/52/8 of72 each;
cross-target Hi E S P. activates20/72. c161 common errors are4/18/29/4, cross0/72;
v30 common errors0/1/0/0, cross0/72; official common/cross both zero. New-model
worst stratum is EN-BR/bare7/9; every command presentation activates. With common
confusion capped at7/288, no threshold on its consumed grid exceeds23/72 target
hits and the best worst-stratum recall is zero. This is a descriptive grid bound,
not permission to substitute a posthoc point.

Continuous81600sample zero prime passes through the frontend/model/decoder;
no transition reset, readiness censoring or discarded prime events. Candidate
five-UINT8 event arithmetic, original v30 arithmetic, native480/512block clocks,
half-threshold rearm and refractory remain explicit. Official DEFAULT mode and
getter, UART/raw-return completeness and source CRCs were preserved in original
QEMU receipts. QEMU classicESP32 is not evidence of P4 numerical equivalence or
resource feasibility. Candidate training completed20000RTX4090updates; CPU replay
evaluated actual deployment INT8 bytes and the unchanged official reference.

First detected positive event minus generated-prefix reference, median:
new 7.9375ms, c161 59.78125ms, v30 0.25ms, official -8.4375ms. Populations differ
(70/60/5/33 detected clips), so these medians do not establish a latency ranking.
New 33/70, c161 14/60 and v30 2/5 first hits precede that reference. Complete signed
per-event records, including all four generation/container/RIR reference clocks,
are retained. These are internal synthesis coordinates, not human last-phoneme
truth; early events were not reclassified as failures or trimmed from recall.

Authoritative aggregate: `results/4090-melo-joint-development-findings-v2.json`.
Full tables, per-record grids and selected event details are preserved as
`4090-melo-development-v2-*` / `4090-melo-official-v2-*` gzip artifacts. Source
generation/native/render/role locks, failed interventions and all original bytes
remain. Original v30 is54440B/SHA256
`04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c`.

Next distinct route is an explicitly different pretrained acoustic encoder and
fixed-waveform-window task, not another threshold or lexical-loss rescue on this
encoder. MatchboxNet original weights strictly load and basic forward/backward
works on4090 with pseudorandom input only. The source-semantics audit and
training-only raw-material plan are complete; no Hey Nicks training, INT8 export,
recognition improvement or P4 claim follows yet. It cannot join this closed
two-model cohort; future qualification requires new unconsumed development and
the still-sealed independent acceptance under a separate prospective protocol.
