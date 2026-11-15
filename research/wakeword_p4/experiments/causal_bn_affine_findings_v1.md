# Fixed-moment BN affine adaptation findings

Four BN gamma/beta pairs trained with original moving moments fixed; original25981parameter scalar readout.23trainables/8fixedmoment tensors/48Adamvariables. Originalinit/sourcebalanced/seed726/Adam1e-5/final-only20000RTX4090updates; no auxiliary or larger head. Originalsession18420 exited0 in1525.615s, report2675f2af. Exportlock1f1f70e2 before session70200 exit0. INT8 60952B SHA256 ba9aff8c2e6b0f8a3807db1286b8b2ca5f43b829c6372b78e5d1a3ebcd3c6c97. Export original and fullwave locks af7a8a0c preceded all three collection sessions14091/11448/46938, all exit0.

Fixed.91796875/smoothing5/81600continuousprime, complete1140inputs, no source-boundary reset. Reuse saved c161 baselines without reinference.

| Consumed development | c161 baseline | BN affine |
|---|---|---|
|Legacy targets / commands|35/36;26/27|34/36;25/27|
|Legacy common false clips|4/288|3/288|
|Jenny targets / commands|33/36;24/27|32/36;24/27|
|Jenny common false clips|44/324|43/324|
|Jenny Nick/Next/Mix/Muse|0/8/25/11|1/5/27/10|
|Joe targets|0/72|0/72|
|Legacy background|0/10.483944826h|0/10.483944826h|
|Jenny background|0/9.614419583h|0/9.614419583h|

Paired saved events: legacy1lost/0gained target,1removed/0newfalse. Jenny1lost/0gained target,5removed/4new/39retained false. Coverage reduced without materially improving lexical confusion; Joe blindness persists. No promotion or threshold/checkpoint/calibration rescue. Fixed BN affine adaptation alone does not solve the observed behavior; not a causal diagnosis of why it fails.

Legacy generated-prefix complete34/36, commands25/27, median143.375ms/p95195.072ms over34hits. Jenny generated-prefix complete24/36, commands16/27,8before-only references; median102.688ms/p95257.666ms over32hits. Timing uses synthetic generation boundaries, not human phonetic truth; early-reference events are descriptive, not an automatic rejection. Background FAR95% upper.285744758/h and.311587428/h; reused exposures are nonadditive and not new independent evidence.

All1140original reports andUINT8traces losslessly preserved499419 bytes. Comparison/event-delta include hashes and saved per-case events. P4 arena/latency unknown. Existing consumed development is not threshold-selection or acceptance evidence. Independent threshold selection/final INT8 qualification/official matched final evaluation remain incomplete; frozenv30 retained.
