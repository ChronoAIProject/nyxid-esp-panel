# v41 preregistration: negative complete-timeline support

Baseline v40, not v38/v39. Actual v40 exported float streaming reproduces103
high negatives in the diagnostic union;83 peaks end after training crop end,
median32frames, beyond v38's14frame jitter. No sole causal claim. v39's fixed
162frame/five-score loss and v40's center-only source coverage both failed.

Change only the24candidate negative windows at slots32..55. CompleteCoverage
must make exactly the same40canonical source/variant draws as v40 using127726.
Positive16candidate slots retain exact original crops. For each selected negative,
use independent RNG137726 to choose an integer end uniformly from150 through
len(raw), inclusive; use raw[end-150:end], normalize clip(raw/25.6,0,26).
Every integer phase allowed, no score mining, acoustic-boundary filters, padding,
new rendering or label changes. Raw histories are pinned Mac uint16 features.
Negatives may contain partial confusion phrases or background-only windows;
this is explicitly a negative temporal-distribution intervention, not full-phrase
quality evidence. Positive partial prefixes are NOT relabeled. Original24British
negative/16British positive/48ambient draws and base RNG remain unchanged.

Keep original128weighted BCE,negativeweight20,teacher60/context150/frozenBN,
reconstructed initialization,seed726,Adam/schedule120k,eval5k,checkpoint rule.
No supplementary loss/head or five-score objective. Representative INT8 data
uses same new sampler. Keep original output architecture/state/export policy.

Before optimization: hash every eligible negative raw file against immutable
records; require uint16[frames,40],len>=150. Audit50batches against independent
v40source draws and independent temporal RNG/crops, verify retainedslots and
all selection bounds; retain endpoint distributions without scoring. Two-step
4090smoke must show finite weights/frozenBN and original streaming INT8 contract.
Then formal120k once, historical full regression/fixed138point gate unchanged.
Reject on zero eligible points. Only passing old-stage candidate may use separately
preregistered untouched SLR83/newVox development versus frozenv30/officialHiESP.
No finals,Mac350,threshold extension or fallback promotion. Frozenv30 retained
unless all original acceptance gates pass. Timing uses generated process bounds,
never phonetic truth or human generalization claims.
