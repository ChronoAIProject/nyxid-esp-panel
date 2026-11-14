# Frozen encoder: completed diagnostic findings

All four original fixed-final 10000-update runs completed; own540 FP32 waveform numerical checks passed. All1488 saved full traces were audited, all552 diagnostic points/events independently verified, and all276 matched deltas retained. Original diagnostic summary is copied byte-identically into results (SHA256 1a6e9b78983ce5963080d48fb01aca5fb21bfef631b36f24334e3d41e8d8a232).

| Target / condition | FAR-qualified points | Negative-only joint points | Complete positives at FAR points | Complete positives at negative-only joint points |
|---|---:|---:|---:|---:|
| Nyx full |138|30|0–35/36|0–14/36|
| Nyx frozen |138|138|0–1/36|0–1/36|
| Willow full |121|121|0/36|0/36|
| Willow frozen |138|138|0/36|0/36|

“Negative-only joint” means FARupper95<=0.5/h and zero common/cross/pre-source events. It is a necessary diagnostic condition, not overall qualification. The comparison field `joint_qualified_points` uses that restricted definition; `rejected_for_no_joint_point: []` does not mean any candidate passed. No new positive-recall gate is introduced retrospectively. These results do not establish a useful detector or satisfy the original promotion requirements; all four are rejected for promotion and no working threshold is selected.

At all138 identical thresholds Nyx frozen-minus-full complete counts range -34..0, common events -35..0; background/cross/pre-source deltas all0. Frozen Nyx command recognition is0/27 at every FAR point; its only possible complete hit is standalone. Willow complete count is0 in both conditions at every point; freezing removes0..4 background events, with common/cross/pre-source deltas0. Reduced negative triggering alongside nearly absent positive triggering is not evidence of improved discrimination. At threshold0.5 Nyx full has35/36 complete positives but35 common events (21 bare,14 command), so its high recall alone is also insufficient.

The pinned upstream frozen speech-encoder approach motivated a distinct test, but this local context225 backbone/head/synthetic-data/fixed-budget recipe did not yield a useful frozen representation detector. This single run per arm does not show that freezing never works or establish which data/representation/optimization factor causes failure. Do not extend budgets, sweep learning rates, select snapshots, recalibrate, or open fresh splits to rescue these runs.

Keep every source/carrier/length/variant group and the original reference curves in the accompanying JSON. Whole-background exposure is10.483944826388889h/model; minimum upper95 is0.2857447576425148/h. This correlated reused Semaine diagnostic is not independent acceptance. Generated-prefix boundaries are synthetic metadata, not human phonetic truth. FP32 numerical agreement is not INT8 qualification. v30 is unchanged; freshdev, independent acceptance and the three historical final sets remain closed. No new P4 execution is requested.

Next research work must address a new, explicit recognition hypothesis supported by training/source evidence rather than another blind numerical-recipe variant. Full results are handed to Mac for independent table verification before interpreting a new intervention.
