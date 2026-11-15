# Nonlinear deployed readout findings

Fixed 3480→16 ReLU→1 sigmoid deployed readout, 78213 parameters; same causal encoder/context309/stride3/frozenBN and source-balanced binary recipe, no auxiliary loss. Original RTX4090 session22247 completed20000 updates, exit0,1403.951s. Training report a19dc80a, export lock9d5027aa, export/result and complete-wave locks d0e90459. One INT8 export115848 bytes SHA2564d81793f10dd68f480aa97be764254fd45fd66464a5155cbefd6a080bed42a4f. Actual P4 arena and latency remain unknown.

All original collection sessions44727/49673/89335 exited0:372legacy,408Jenny,360Joe. Fixed threshold0.91796875/smoothing5/81600continuousprime; no reset at source boundary, baseline saved traces reused.

| Consumed development | c161 binary baseline | Nonlinear deployed readout |
|---|---|---|
| Legacy target / commands |35/36;26/27|35/36;26/27|
| Legacy common false clips |4/288|21/288|
| Jenny target / commands |33/36;24/27|30/36;22/27|
| Jenny common false clips |44/324|31/324|
| Jenny Next/Mix/Muse |8/25/11|6/20/5|
| Joe target |0/72|0/72|
| Legacy background |0/10.483944826h|0/10.483944826h|
| Jenny background |0/9.614419583h|1/9.614419583h|

Jenny generated-prefix complete reference22/36, commands14/27, with8 before-only cases. These are generated-boundary descriptions, not human phonetic truth or an automatic early-event rejection. Legacy generated-prefix median123ms/p95226.175ms; Jenny96.406ms/p95260.272ms over observed hits, not matched human latency claims. Background FAR one-sided95% upper0.285744758/h and0.493411430/h respectively; reused exposure is not independent/additive.

Paired saved events: legacy1lost/1gained target and17new/0removed false; Jenny3lost/0gained target and13removed/0new false. All31Jenny false cases remain baseline false cases. Overall this changes confusion tradeoffs, does not resolve Joe blindness, and adds a background event while greatly worsening legacy confusion. Larger nonlinear deployed readout alone is not supported as an upgrade; no threshold/checkpoint/calibration/width rescue. No causal claim about encoder capacity follows.

All1140original reports andUINT8traces preserved losslessly476522 bytes. Comparison and paired event-delta JSON contain hashed inputs and actual per-case events. Consumed development only; independent threshold selection/final qualification/official matched final evaluation remain incomplete. Frozen v30 retained.
