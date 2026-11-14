# Speaker coverage: actual INT8 behavior is not usable

The single matched narrow6 versus broad30 experiment completed 20000 updates
per arm from identical preserved initialization on RTX4090. Broad was the sole
candidate; narrow was a diagnostic control. Both fixed-final models were
exported once. Whole native/render/feature provenance and full training receipts
remain available; no checkpoints, seeds, LR or calibration packs were selected
to rescue these outcomes.

All 744 actual INT8 consumed-input traces (372/model, two fresh interpreters,
complete startup/tail and 10.483944826388889 hours of continuous negative audio
per model) passed repeat/code/probability/clock checks. Original scoring exited0;
independent event/count/exposure/latency reduction verified all276 points.
These are consumed diagnostics, not fresh development or acceptance.

| Actual INT8 model | FAR upper95 <=.5/h grid points | Complete positives at those points | Standalone at those points | Points also passing negative/pre-source requirements |
| --- | ---: | --- | --- | ---: |
| narrow | 0/138 | none | none | 0 |
| broad | 5/138 | 4–5/36 | 0/9 | 0 |

Minimum measured Poisson FAR upper95 is .8730987408096108/h for narrow and
.2857447576425148/h for broad. Varying-threshold ranges are descriptive, not a
chosen threshold or a claim of matched-FAR superiority. Full threshold curves,
matched broad-minus-narrow deltas and historical same-input reference curves
are retained in `4090-multispeaker-formal-comparison-v1.json`.

Every threshold has positive pre-source events: narrow24–36, broad9–24. Saved
event reduction over all synthetic classes at the already registered maximum
threshold .99609375 finds narrow240/360 clips with240 events at sample7040
(.44s), broad80/360 clips with80 events at samples11840–13760 (.74–.86s).
These events precede the serialized source-placement boundary, so they cannot
all be explained as hearing the near-word phrase. This is behavioral evidence;
it does not identify an acoustic, numerical or training cause. It warrants
reviewing startup/state coverage against historical failures before declaring
a distinct next training hypothesis, rather than another speaker/LR sweep.

Supplementary numerical screen: narrow135/135 passed, broad133/135 passed.
Broad requests48393 and106323 exceeded the original five-step p99 score-error
limit .1 (worst .10760704588890069). Other recorded worst broad mean/max errors
were .011031162035426898/.1166711211204529. These diagnostic failures are
preserved; the recognition-first steering did not make them a prerequisite for
actual INT8 behavior measurement. No further float comparison was launched.

The speaker-coverage intervention does not yield a usable candidate. No
threshold is selected, no fresh development or protected acceptance is opened
to rescue it, and frozen v30 is retained. Narrow cannot become a fallback.
The active research objective remains incomplete: a usable synthetic-protocol
model, fresh development/acceptance and official-HiESP/frozen-v30 same-PCM
behavioral comparison still need to be achieved. No human listening or
phonetic-reference prerequisite is introduced and no human/P4 parity is claimed.
