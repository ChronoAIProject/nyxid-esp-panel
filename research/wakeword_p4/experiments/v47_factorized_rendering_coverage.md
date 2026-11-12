# v47 proposed factorized rendering training coverage

Freeze before pack assembly/training. v46 remains rejected; v30 retained.
All-source frozen-model factorial found noise15-only reduces positive support
1849→1382/2016 versus RIR-only1846, and negatives0→6 versusRIR-only1.
Upper input clipping is zero; v22 affine-BN already failed under its old recipe.
Hypothesis: exposing the existing source/phrase-balanced sampler to separately
rendered noise and RIR conditions can improve robustness beyond joint-only training.
This does not assert it solves four clean development errors or conversational FAR.

Only expand balanced pack from original3 variants per speed to5:
quiet,noise25,combinedRIRnoise15,noise15-only,RIR-only. Preserve every original
variant and all8064 sources; no filtering, mining, relabeling, new identities or
modified raw audio. Total40320variants,672 IDs×4phrases×3speeds×5conditions;
each ID/phrase15variants. Original controls audit must remain exact. New pack
records original byte/hash links and source commits; retain complete raw arrays.
The controls were already used in descriptive training-source diagnostics; they
are not unseen data and must never be presented as independent validation.

Proposed v47 versus v46: uniform15 instead of9 choices within ID/phrase; same
207726source/217726end seeds,20 balanced slots8..15/44..55,original mixed end
policy,untouched108slots and all inherited RNG draws. Changing cardinality changes
selected source/speed draws and augmentation mass: original3 conditions each1/5
instead1/3; new2 each1/5. Not additional batch mass, not isolated noise-only effect.
All teacher60/context150/frozenBN/init/batch128/weights1:20/ambient48,Adam schedule,
120k/every5k/checkpoint/calibration rules preserved. No threshold/grid rescue.

Before formal training: independently authenticate40320raw arrays/crops/labels,
15variant grouping and exact original subset,source identity partition; independently
replay50batches/source+end RNG,untouched slots/states. GPU2step smoke must pass
finiteweights/frozenBN/int8[1,3,40]→uint8[1,1]. Pin exactpack/reporthashes and executable
recipe before120k. No run until those concrete prechecks complete.

Postprocessor authenticates exact recipe then complete historical regression,
oldKokoro/VCTK/Vox and fixed138jointpoints. Failure rejects before untouched dev/final;
retain original evidence,do not expand exposure/grid or automatically startv48.
If old-stage passes, still require originalquality, untoucheddevelopment calibration,
final gates and samePCMofficial comparison before promotion. No human/board prerequisite.
