# v42 preregistration: complete-container positive temporal coverage

Freeze before integration/optimization. Baseline v41, whose negative full-timeline
sampling reduced training fullINT8 high negatives35->0 at.99609375 but failed
all138old-stage points. Actualfloat380olddev clip decisions match INT8,including
13negative failures:quantization is not necessary for these observed confusions.
Savedsequencefloat positive full-container five-position hits at highthreshold
fell2468->1374/4602;not proof of sole cause or improved generalization if repaired.

Change only16candidate positive windows,slots0..15. Preserve v41canonical
source/variant draws seed127726 and all24candidate negative endpoint draws
seed137726 exactly. Independent positive temporal RNG147726 chooses uniformly
integer end from original record.feature_crop_end through350 inclusive, bounded
by raw length. Use raw[end-150:end],clip(raw/25.6,0,26). This exactly follows the
immutable Mac pack conservative sample ledger:begin=(end-150)*160<=32000,
stop=(end+1)*160>=32000+signal_samples. Signal includes whole source/fullRIR.
This ledger does not certify receptive-field/phonetic support. All variants retain
mass regardless of interval width. No score/boundary filtering,no padding,no
relabeling or audio regeneration. Preserve original center crop among choices.
All phases allowed. Original16British positives/24British negatives/48ambient
unchanged. No positive prefixes labeled negative. v41negative partial windows
remain negative as preregistered. Distinct from v38British-only offset0..14 and
v39local162frame five-probability supplementary loss; no extra objective here.

Keep v41all other training/export settings:original128weighted BCE,negative20,
teacher60/context150,reconstructedinit,seed726,frozenBN,Adam/schedule120k,
eval5k/historical checkpoint rule. RepresentativeINT8 uses same new distribution.
Before formal training:authenticate all eligible raw arrays,source/crop ledger,
50independent batches and unchanged source/negative RNG and retainedslots;
2step4090finite/frozenBN/originalstreaming INT8contract. One formal120krun only.
Then same historical regression/fixed138gate. No newdev if no eligible point;
no threshold extension,hour merging,event removal or fallback promotion.

Old-source mechanical endpoint failures/native-float provenance gaps remain;
this is a controlled research diagnostic and not evidence all synthesis quality
gates pass. Frozenv30 remains;new/final untouched. No promotion without original
quality,positive/negative,quantization/FAR95<=.5/h and separately frozen final
requirements. Results limited to synthetic protocol,not human generalization.
