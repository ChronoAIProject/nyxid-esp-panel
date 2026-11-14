# Bare-positive causal correction: fewer false wakes, connected-command misses remain

One fixed RTX4090 seed726/20000-update run from the same original initialization
completed naturally (session85474, exit0,865.738s). Only positive sampling changed
to existing full `Hey Nicks.` / `Hey Nicks!` sources. Synthetic negatives,
ambient, causal graph, variable onset and optimizer remained fixed. This tests
removing whole-command completion positive supervision; no new TTS, audio crop,
common-carrier continuation, checkpoint selection or numerical alignment audit.

The single final INT8 export (session39549, exit0) is60952bytes,
SHA256 `b9f9b283cca398ef6c0bb70c13759ad5221bcefd288cca640911103cbd4fc0db`.
Actual existing-primed collection31381 naturally exited0,372full inputs and
10.483944826388889h continuous backgrounds. Scorer74129 exited0 with138points.
The same81600zeroPCM prime is continuous through frontend/model/event state;
all real firstphrases and complete input finishes are retained. Prior causal
and originalv30 actual-primed results are reused without duplicate inference.

At fixed reference threshold0.91796875/smoothing5:

| Behavior | Bare-positive | Previous causal | Original v30 |
| --- | ---: | ---: | ---: |
| Complete target |28/36|32/36|25/36|
| Bare |9/9|9/9|8/9|
| Commands |19/27|23/27|17/27|
| Light command |1/9|5/9|0/9|
| Time command |9/9|9/9|9/9|
| Timer command |9/9|9/9|8/9|
| Common false clips /288 |1|9|2|
| Cross-target false clips /36 |0|0|0|
| Background events |0|1|0|
| FAR one-sided95% upper /h |0.285745|0.452489|0.285745|

New bare hit-conditioned median71.75ms; command median183ms. The sole new
light hit is222.6875ms after generated-prefix support, versus previous causal
five hits' median1272.6875ms. **Eight new light misses remain**: comparing only
hit latency would hide the regression. Time/timer medians183/166.5625ms.
These are generated support coordinates and480sample app-return clocks, not
human phoneme times. The new common false event is variant2 of consumed slow
`v2-t09-c2-s2-r00`, Muse/time:335ms after generated prefix and1058.1875ms before
whole-container end. All errors and misses remain in the published evidence.

All138secondary gridpoints meet the background FAR bound in this consumed
exposure; maximum complete target30/36. Twenty-six points have zero common/cross
false events here, with complete22–28/36. This is descriptive failure analysis,
not threshold selection or a universal perfect-model gate. The mature Muse
reference also has near-word errors; neither that fact nor zero diagnostic
errors establishes Nyx qualification.

This minimal change reduces observed false wakes and removes the long-delayed
light hits, but leaves substantial light-command misses and lower overall
recall than previous causal. It does not prove a sole cause or solve connected
speech. Do not promote; original frozenv30 remains. No fresh development or
acceptance has been opened. Independent multi-voice threshold development,
acceptance and officialHiESP same-protocol comparison are still incomplete.

All original report/138point JSON bytes and372rawcode streams are losslessly
published in690740bytes, with per-file SHA receipts. No duplicate PCM/features.
See `../results/4090-causal-bare-positive-fixed-reference-behavior-v1.json` and
`../results/4090-causal-bare-positive-primed-evidence-manifest-v1.json`.
