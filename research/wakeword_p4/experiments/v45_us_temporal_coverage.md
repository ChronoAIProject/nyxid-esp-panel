# v45: time coverage of the existing US training cohort only

Evidence: frozen v44 fits the exact newUS negative supervision crops (0/243
float highs at .99609375), yet four new-training variants have fullstream
INT8 highs. Enumerating every legal150frame float window for these four finds
6/15/8/8 high ends;6/13/8/8 are complete-container ends. Peak window scores
>.998 although original crop scores<.000003. Two high ends in one variant are
not complete-container windows. Actualfloat and INT8 streaming are distinct;
these observations do not establish a sole cause or claim fixing training fit
will solve old-development identity generalization. Old-development still has
11shared actualfloat/INT8 failures; original checkpoint differences remain.

Hypothesis: applying conservative positive complete-container endpoint variation
and mixed complete/full-timeline negative endpoint coverage to the newUS cohort
reduces its untrained time-position responses. NewUS in v44 was fixed-crop,
while inherited source cohorts already use temporal coverage. This is not a
rerun of an old-cohort temporal intervention, and not a source-mass increase.
Only time sampling of the existing20US slots changes. No selected witness is
added, weighted, relabeled or specially sampled; all324variants remain eligible
under their original equal preset/phrase/variant source distribution.

Baseline v44: original main/British/complete/v43timeline/container draw seeds,
US source seed167726,8positive slots0..7 and12negative slots32..43, pack SHA
4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da
all unchanged. Consume every original source/variant draw. New independent
numpy default_rng seed177726, draw once per selectedUSexample in fixed order:
-8positive slots: uniform integer end crop_end..min(350,len(raw)).
-first6USnegative slots32..37: same uniform complete-container ends.
-last6USnegative slots38..43: uniform integer end150..len(raw).
No other endpoints/windows, feature alignment changes or phase-score selection.
Save all original raw arrays; authenticate their shape/dtype/data/file hashes.
For every324variant independently enumerate legalcomplete ends and verify exact
sample ledger [end-150,end+1]*160 includes whole source and fullRIRtail. No
phonetic truth/receptive-field claim. Original trained crop must be the lowend.

All remaining108batch slots, source/variant probabilities, ambient slots,
labels/weights, architecture/teacher60/context150/frozenBN/init/optimizer,
120k/every5k checkpoint policy and original representative calibration logic
unchanged. New calibration uses same newUS time mixture. No loss/head/mining,
voice generation, background exposure extension or gate changes. A temporal
distribution intervention may affect calibration/checkpoint; no isolated causal
claim from final v44/v45 numbers.

Before fitting: audit every324legal endpoint set/raw/lowcrop; independently
replay50batches of source and new temporal choices; exact originalsource RNG and
untouched slots. 4090two-step finiteweights/frozenBN/streamingINT8 smoke. Only
then one120k formal run. Original138point old-stage joint rejection gate and
all original quality/INT8/positive-negative/FAR upper95<=.5/h gates unchanged.
Failedold-stage means reject before unseen/final access; no threshold extension
or furthersource mass tuning. Originalinherited native-quality gaps persist,
so temporal success cannot certify all-source mechanical compliance. V30 stays
frozen unless all independent acceptance requirements pass; no automatic
promotion, board rollout or human recording prerequisite.
