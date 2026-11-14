# Same-recipe Willow/Nyx evidence inventory v1

Prepared in response to Issue13 comment6874038949. Existing records only:
no training, export, model/frontend inference, fresh scoring or threshold choice.
Official full calibration is still pending; a quantitative official-vs-custom
same-word gap is NOT yet established from a complete audited official report.

## Model families and identities

| Family | Included models | Source and recipe | Known limits |
|---|---|---|---|
| Original broad float | Willow and Nyx original best.weights.h5 | Original broad formal, then immutable equalized H5 receipt | Historical best checkpoint designation belongs to original training, not new independent selection; equalization is a distinct representation |
| Paired formal | Both words × float_control/integer_qat, all four final10000 repeat0 | 4090-hey-broad-paired-qat-formal-execution/result-v1.json | Fresh Adam lr1e-5 beta .9/.999 eps1e-7, frozen BN, same original sampler, repeat1; integer surrogate is not ordinary FP32 inference |
| Bounded consistency | Both words × raw_logit_control/bounded_logit_consistency, all four final10000 | 4090-hey-broad-bounded-consistency-formal-result-v1.json | Dual raw BCE plus selected consistency; bounded nominal coordinate differs from original raw sigmoid; all complete numerical conditions rejected |
| One-way consistency | Both words × twoway_consistency/oneway_consistency, all four final10000 | 4090-hey-broad-oneway-consistency-formal-result-v1.json | Only C float-target gradient stops; shared weights still move target. Not independent/EMA teacher; repeat1. Full540 rejected |

FP32 scope d652049b fixes all four paired final H5, with original raw-sigmoid
reconstruction, not bounded output. Exact original H5 paths/bytes/SHA and recipe
source hashes are recorded in 4090-fp32-original-paired-route-prospective-v1.json.
They are diagnostic conversion controls, not qualified deployment candidates.
This inventory does not choose a best family, checkpoint, target or threshold.

## What existing records establish

Original old-fixed-grid collection/metrics/independent-audit-v2 cover nine paths
×138 points with complete saved event ledgers. Mac independently read original
NPZ (59c1946), cross-checking44712 positive rows/178848 generated-boundary rows.
Old Willow float-control float any-event is only7–8/36 across fixed points;
UINT8 any-event7/36. Old Willow integer-QAT float and UINT8 any-event0/36.
Thus most misses are absence of any event, not complete-boundary filtering alone.
Old Nyx float-control UINT8 macro .861111–.972222 and FAR upper .452489/h
on these consumed diagnostic records; this does not qualify a new Nyx candidate,
prove a matched official gap or establish Willow intrinsic word difficulty.
Different models/targets cannot be promoted by comparing their best endpoints.

The later bounded and one-way families are NOT these old recognition models.
Their numerical rejections cannot stand in for recognition measurements, and
old recognition metrics cannot rescue their numerical failures. One-way full540
passes125/126 (Willow two/one-way) and126/127 (Nyx two/one-way), each of135.
No condition meets both-target complete270 qualification.

## Effective inputs, labels and coverage: evidence boundaries

Current broad-family sampler uses128×225×40 batches,32 positives/96 negatives,
positive weight1/negative20; original prefix-negative replacement accounting
must distinguish retained ordinary negatives, discarded draws and prefix slots.
Context support enumeration establishes positive ends keyword+3..27 and Piper/
Kokoro prefix ends keyword−35..−15; Qwen has no such prefix role. This is the
actual metadata labeling rule, not phonetic truth or proof every desired full
utterance window was trained positive. No automatic relabeling follows.

The authenticated10000-batch replay reported in GPU_RESEARCH.md/31165c13 is
specific to original bounded formal traces: each target320000 positives,
320000 retained ordinary negatives,160000 discarded ordinary draws and160000
prefix negatives. Do not retroactively claim this derived replay is a historical
saved source ledger or that it audits every earlier paired model. Batches shared
across target inventories/RNG do not establish identical audio. Engine draw
balance does not establish per-source coverage or heldout independence.

Saved old bounded final forward and gradients authenticate selected training
batches and show rare nominal float/FC disagreement on training support. They
are static contexts, not full utterance recognition. They do not prove the
original paired Willow misses share that cause. Uniform batch C can coexist
with large individual nominal errors; prior tail-risk failures prohibit treating
that association alone as justification for blind tail-loss retraining.

## Full-stream semantics and unresolved comparison

Original raw sigmoid, bounded sigmoid and deployed UINT8 posthead are distinct
observables. Preserve exact source/model/head identity when quoting results.
Six fresh-zero states per whole waveform; stride3, all startup frames, final
padding and unclamped clocks retained. Nominal225 and phase224+zero references
are different; never shift runtime to force alignment. Generated dry/full-RIR
boundaries are synthetic references, not human phonetic endpoints.

Before attributing official Willow superiority to data or training, complete
original9672 grid and original audit. Compare all registered actual getter points
at common FAR requirements, not equal numeric threshold values or cherry-picked
points. This old diagnostic grid is consumed evidence, not fresh acceptance.
A future valid candidate still needs its own source-quality and numerical
admission, untouched development workpoint, same-PCM reference and independent
synthetic acceptance FAR upper95<=.5/h. FP32 runtime feasibility alone fixes none
of the known recognition misses. P4 runtime/whole-panel evidence is separate;
a board remains unnecessary for continuing synthetic INT8 research.
