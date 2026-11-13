# Recover overwritten CompleteCoverage draws: matched experiment v1

Hypothesis: US/factorized replacements erase the first8 positive and12
negative CompleteCoverage draws before optimization. Restoring exactly these
20 already sampled rows increases complete-container source exposure and may
reduce historical cross-source confusion. This does not prove that British
pack exclusions caused failures; this intervention differs from prior temporal
loss/noise/slot recipes by preserving these actually erased draws.

Control: patched trainer with recovery disabled. Treatment: same trainer with
--recover-erased-complete-coverage. Both use fixed paired noise; no per_draw
noise or extra loss. The enabled branch replaces British rows16..23/56..67,
so conclusions concern this mixture change, not an increase in batch size.
Source/voice identity balance of the full mixture can change intentionally.
All sampler calls/RNG, main128/pair40, labels/weights, initialization seed726,
coefficient1, frozen BN, deterministic FP32/noTF32, original120000/5000
schedule, checkpoint ranking and PTQ remain unchanged. One fresh120k per arm;
no old trained control, repeated seed search or posthoc checkpoint choice.

Prerequisites: exact original/default-off/enabled actual two-update comparison
on CPU and RTX4090. Default-off must reproduce original exactly; recovery must
affect only the20 specified rows and leave108/main plus pair40, labels,
weights, permutation and captured RNG unchanged. Saved draws must reach the
real optimizer. Finite updates and unchanged BN required. Preserve failures.
Pin executable sources, full training inputs and configs before any formal.

Run control then recovered sequentially; preserve original handles, never
restart because observation times out. Authenticate full120k ordered paired
source schedule and pair input hashes equal between arms, main input hashes
different, identical initial inventory, finite consecutive losses and export
contract. Existing source geometry/PCM pins remain prerequisites.

Evaluate BOTH selected exports with the unchanged historical138-threshold,
smoothing5/event-latch and complete old background protocol. It is consumed
exploratory regression only. Report null eligible thresholds and failures.
No gate/grid extension, source mining or background rescue. Prior consumed
SLR83/dev2/dev3 and all3finals remain excluded from training and new selection.
A historical survivor is not a promotion.

Before scoring any new-source development: separately preregister mechanical
synthetic quality/provenance, disjoint training/development/acceptance roles,
full source PCM/available phones/timing, no score-based regeneration, fixed
threshold/tie rules and matched official HiESP/v30 same-PCM streaming. Unknown
physical speaker/pretraining overlap must be disclosed, never inferred away
from dataset names. Independent acceptance must pass original positive and
confusion gates plus FAR one-sided95 upper<=0.5/h. No human hearing/recording
prerequisite; generation boundaries are synthetic latency references only.
v30 remains frozen; this experiment alone cannot satisfy final acceptance.
