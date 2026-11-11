# v37: replace phrase supervision with mechanically complete synthetic sources

Hypothesis: the v35/v36 supplement (.05 positive BCE mass) is dominated by the
old phrase batch (.25 positive mass), whose crops are selected from proxy
endpoints. Replace the entire phrase part rather than append another weak
loss or auxiliary task. This does not claim old labels are acoustically wrong:
original PCM includes silence, and whole-source support is not phonetic truth.

Use the pinned british-training-features-v1 pack (SHA
c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096),
1002 complete-source crops from424 training sources. Keep all270 unsupported
augmentation attempts excluded and reported; no source trimming/relabeling.
Replace the original32 positive and48 confusing-negative slots with32/48 full
source examples: equal kokoro/alba/cori checkpoint groups, equal voices within
group/phrase, equal Nick/Next/Mix negative phrase choice, uniform rows. Seed97726.
These are checkpoint groups, not certified independent people. Leave48 ambient
slots, stratified ambient distribution and negative weight20 intact. Batch128,
no extra BCE/head, prefix0 because all48 phrase-negative slots now come from
explicit generated confusing phrases. Preserve original main RNG draws before
replacement, then shuffle identically; independent full-source RNG. Calibrate
INT8 with the same new training distribution, no development/final audio.

Teacher60/context150/frozenBN, original reconstructed initialization, optimizer,
120k steps/every5k historical checkpoint selection unchanged. No hard-window
mining. This differs from v32 (old phrase data with prefix removed), v33 (old
phrase data with prefix loss masked), v34 (extra auxiliary head), and v35/v36
(main phrase labels retained with supplementary BCE). The intervention is the
entire phrase supervision distribution, not an isolated prefix ablation.

First verify sampler determinism, phrase labels/source groups, original ambient
preservation, finite weights/frozenBN and streaming INT8 contract in two-step
4090 smoke. Only then run120k. Existing declared development is exploratory
regression and must retain failures; no grid/exposure expansion to rescue.
If no eligible point, reject without spending the unused SLR83/dev2-dev3 stage.
If eligible, use that already preregistered new development stage for threshold
selection and same-PCM official comparison, then separately frozen final
acceptance. No automatic promotion; retain v30 unless all original gates pass.
