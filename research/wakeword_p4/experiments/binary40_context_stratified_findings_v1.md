# Stratified training frontend/cache interaction

Fixed prelock3f09a4ec: all30 original training engine/speaker strata; one lexicographically first source per bare, time and light phrase (90sources), same-voice first Hey Nick negative as history. No score-based selection. Original command-prefix references are bound by SHA256, not treated as phonetic truth. RTX4090 original Binary40 final20000weights, no optimizer. Actual session54726 exit0; all360 crossing metrics independently recomputed from preserved complete probability arrays. Physical diagonal full/split probabilities agree exactly.

| Training phrase | Silence frontend / silence cache | Silence frontend / speech cache | Speech frontend / silence cache | Speech frontend / speech cache |
|---|---:|---:|---:|---:|
| Bare /30 |30|30|29|26|
| Time command /30 |30|30|30|30|
| Light command /30 |30|30|30|30|

Both prefix-limited and whole-container crossing counts coincide. In the bare group, index27 (GB aru) fails with speech frontend regardless of cache. Indices51,54,69 (US arctic) fail only in the combined speech frontend/cache arm. A silence frontend with speech cache loses none. This demonstrates a crossing-level history effect on these fixed training native sources, including a frontend/cache interaction; neither a universally defective frontend nor a cache-only failure follows. It does not establish the mechanism of every consumed development miss or population-level effect. The off-diagonal cache transfers are artificial interventions, not deployment resets or proposed policy changes.

Gap is8160samples after480-aligned negative history, with81600zeroPCM before that history; equal-length all-zero prefix is the control. Native targets differ from augmented cached training presentations. Score window uses original support ledger through automatic prefix+4320samples, not acoustic/phonetic alignment truth. GPUfloat findings need separate deployment-INT8 verification before inference about deployment arithmetic. No false-positive rate, human/P4 acceptance or first-command retention claim.

The prior first12 feasibility diagnostic was homogeneous and missed this effect; it must not substitute for stratified evidence. Prior C161 paired speech-history training already improves short-gap consumption but trades British bare coverage and daily six-cups conflicts. A new training proposal must preserve ordinary standalone coverage explicitly and distinguish its objective from that prior recipe; merely rerunning context augmentation is not selected. Next check historical invariance/distillation interventions and confirm this fixed stratified effect with deployment INT8 before locking a distinct paired objective. No threshold/frontend/reset changes or failed-clip additions are selected.

Artifacts: `4090-binary40-context-stratified-{plan,original,review}-v1.json`, complete trace NPZ and original compressed log. Baseline C161 remains fixed.
