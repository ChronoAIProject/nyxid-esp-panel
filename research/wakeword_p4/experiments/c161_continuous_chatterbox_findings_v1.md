# Continuous mixed-session development diagnostic

The preregistered a92565e4 protocol completed with exit 0. No training or threshold changes. Existing 36 builtin Chatterbox development clips were reused, ordered Nyx / everyday negative / Willow for each index. Each gap condition has one 81600-sample zero prime and one continuous frontend, model state and event policy. Native480 return clocks and raw UINT8 traces are archived; archived members were byte-verified and the event policy independently rescored from saved codes.

| Inter-clip gap | C161 Nyx containers with events | bc01 Nyx containers with events |
|---|---:|---:|
| 0.5 seconds | 0/12 | 4/12 |
| 2 seconds | 2/12 | 10/12 |

Neither model produced events in the 12 everyday-negative containers, 12 Willow containers, prime, gaps or final tail, in either condition. These are block-return container counts, not phonetic truth or independent per-utterance detection accuracy. The same inputs are reused across conditions; do not add denominators as independent exposure or calculate real-world FAR.

Previously isolated, separately primed playback yielded C161 5/12 and bc01 10/12 on these Nyx clips. That protocol differs materially. Short gaps in mixed continuous playback now expose substantial sensitivity, including bc01 falling to four event-bearing target containers. This does not establish whether acoustic carryover, model state, or event refractory/rearm dominates. It also does not measure retained command PCM or prove P4 behavior.

C161 remains the baseline; bc01 remains experimental with known British bare-wake regression and a daily six-cups conflict. Next diagnostic should separate event-policy suppression from raw score changes using these already consumed session traces, without resetting or retuning the actual scored policy. Mac should include continuous mixed speech and short intervals in the fixed-profile P4 comparison.

## Saved-score review

Review of every target container's full five-step UINT8 mean found **zero above-threshold containers without an emitted event** for either model/condition. Thus the missing target-container events correspond to below-threshold scores on these fixed traces, rather than event-policy suppression of an above-threshold crossing inside the container. This narrows the next investigation toward continuous acoustic/frontend/model context; it does not isolate any one mechanism. No score or threshold was changed. Per-container maxima and counts are in `4090-continuous-chatterbox-score-review-v1.json`.

## Phase-matched silence-history counterfactual

Protocol 596ba445 was pushed before execution; all 24 target/phase conditions and both fixed models completed with exit 0. Target waveform bytes are unchanged. Each target's original session start modulo480 is retained using 81600 + start%480 leading zero samples, followed by the complete clip and 32000 zero samples. This preserves frontend hop and model call phase while removing preceding speech jointly from frontend and model history. It is a diagnostic counterfactual, not a deployment reset policy. Event clocks and complete raw traces are retained and archived members and recomputed events verified.

| Gap/phase schedule | Model | Original continuous target containers | Silence-history target containers | Gained / lost |
|---|---|---:|---:|---:|
|0.5 seconds|C161|0/12|4/12|4 / 0|
|0.5 seconds|bc01|4/12|11/12|7 / 0|
|2 seconds|C161|2/12|4/12|2 / 0|
|2 seconds|bc01|10/12|11/12|1 / 0|

These paired results support a preceding-audio effect beyond target start phase alone. They do not isolate MicroFrontend adaptation versus model context, and do not prove an exact speech duration or reset mechanism. Comparison between the two silence-history columns also changes phase schedules, so the matching aggregate counts do not demonstrate phase invariance. No tuning, training or fresh/independent exposure is claimed. Baseline remains C161. Next step is a locked frontend-versus-model history intervention on the consumed traces, before choosing a training remedy.

## Frontend/model history feature intervention

Preregistered e635ef77 completed exit 0, 24 target/phase cases x2 models x4 conditions. Reconstructed session PCM hashes match; continuous-control raw UINT8 outputs reproduce the saved original traces byte for byte. All 192 trace members were verified and saved event steps/clocks rescored. Splice point is the first model call whose unrounded support exceeds target onset. The group can straddle onset; this is a feature intervention, not a physical reset or perfectly separable causal decomposition.

|Gap|Model|Continuous|All-silence history|Continuous prefix + silence-history target features|Silence prefix + continuous-history target features|
|---|---|---:|---:|---:|---:|
|0.5 s|C161|0/12|4/12|5/12|0/12|
|0.5 s|bc01|4/12|11/12|8/12|2/12|
|2 s|C161|2/12|4/12|4/12|2/12|
|2 s|bc01|10/12|11/12|11/12|10/12|

Counts are target containers with an event under the unchanged full-session event policy. At 2 seconds, supplying silence-history target features recovers the all-silence aggregate with either prefix, while replacing only the prefix does not recover events. At 0.5 seconds, frontend-conditioned target features remain important, but prefix interactions are evident (C161 5 versus 4; bc01 8 versus 11, and 2 versus 4). Aggregate equality is not proof of identical paired behavior. These results support frontend-history mismatch as a major investigation direction, not an exclusive cause; model context and boundary interactions cannot be discarded.

A training remedy should expose training-only positives and negatives to actual continuous frontend history, rather than merely concatenate cached per-utterance features or reset on unknown boundaries. First audit existing training cache generation and masks for such a mismatch, then preregister an intervention using training-only sources. No consumed builtin development audio may enter training. No threshold or deployed frontend/reset policy changed; C161 remains baseline and bc01 experimental. This experiment provides no new human/P4 qualification or first-command PCM-retention evidence.
