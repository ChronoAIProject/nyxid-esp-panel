# Existing nearword negatives improve practical recognition at the fixed profile

Original RTX4090 run75579/PID310458/start1543119 naturally exited0:
20000updates/875.759s, same original initialization/freshAdam1e-5/seed726.
Timely automatic command-prefix supervision and bare labels were retained.
24of48 synthetic-negative slots use all288 existing training-source negative
waveforms, equally by engine;24old negatives/32positives/48ambient remain.
No new TTS, PCM rendering, crop, score filtering or threshold adjustment.
The reused diagnostic examples are training data henceforth, never independent
acceptance. Only two added source voices is a material coverage limitation.

One original builtinINT8 export98087/PID314132/start1633293 exited0:
60952bytes, SHA256
`9307e6c13b7bc269f2566f111acdddcd8a2fedac569ce9f5c35b4e9fd030a7d6`.
Checks covered file integrity, interface/reset and one functional replay.
No repeat-score equality, saved-array numerical audit or FP32 comparison gate.

Original collector13445/PID314563/start1636705 exited0/all372whole inputs.
Scoring/comparison/publication session5660 exited0. Same81600continuousprime,
all firstphrases/complete finishes and10.483944826388889h backgrounds. Only the
fixed0.91796875/smoothing5 profile was scored; no new grid or threshold tuning.
All previous comparison results were reused without duplicate inference.

| Actual fixed-profile behavior | Existing-nearword | Prior aligned-prefix | Frozen v30 |
| --- | ---: | ---: | ---: |
| Target any-after-source |34/36|33/36|25/36|
| Bare any-after-source |9/9|9/9|8/9|
| Commands |25/27|24/27|17/27|
| Light |7/9|6/9|0/9|
| Time |9/9|9/9|9/9|
| Timer |9/9|9/9|8/9|
| Common false clips /288 |3|17|2|
| Cross-target false /36 |0|0|0|
| Background events |0|0|0|
| FAR one-sided95% upper /h |0.285745|0.285745|0.285745|

Light hit-conditioned median162.6875ms vs prior132.6875ms; command median139.5ms,
all25detected commands within300ms. This is a small timing cost alongside fewer
misses/errors, not a claim that every dimension dominates. Two light misses are
quiet/fullRIR variants of one fast native `v2-t01-c1-s0-r00`. Three common errors
are all variants of one `Hey next, turn on the light.` native
`v2-t07-c1-s1-r00`. Preserve every variant; these are not independent sources.

At/after-generated-prefix hits32/36, with two bare events at-18.25ms relative
to generated support; bare any9/9. These generated coordinates are not phonetic
truth and early generated-end events alone are not failures. Signed offsets,
misses and all errors remain explicit. No success window or model was edited.

This consumed development result supports handing the new experimental model to
Mac for practical P4 feedback. It does not establish final independent acceptance
or replace frozenv30. Mac already reports actual operation of the prior original
aligned-prefix model (Issue13/6886507611): allocation/listening, mean1.321ms,
three short connect/hangup/rearm cycles. Those observations are for the prior
SHA3ea7b067… model, not this new model; no hardware equivalence inferred.
No human recording/listening is required to continue this synthetic research.

Original reports/one-point JSON/372rawcodes are preserved losslessly in150559B
plus SHA receipts. No duplicated audio/features. See
`../results/4090-causal-existing-nearword-fixed-reference-behavior-v1.json` and
`../results/4090-causal-existing-nearword-primed-evidence-manifest-v1.json`.
Independent multi-voice development/acceptance and officialHiESP same-protocol
comparison remain incomplete; no new perfect-zero-error or numerical gate.
