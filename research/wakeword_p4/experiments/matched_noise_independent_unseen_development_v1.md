# Reserved development validation of the independent-noise candidate

Before any reserved-source scores, select only the completed independent-noise
formal model SHA256 83cc9b613062cc706c67736aa6ae500c89d69bbd50546137d211ecf281c00cca.
The common-noise arm failed the unchanged historical joint gate and is excluded.
No training, coefficient fitting, regenerated sources or candidate replacement
is permitted within this stage.

Inherit experiments/slr83_new_development_v1.md unchanged: the 90 full PCM
presentations from two OpenSLR83 Piper presets and three speeds/conditions,
plus the complete 86 dev2/dev3 recordings (8.487218975694445 hours). These are
reserved development, not final acceptance. Historical phone/VCTK/dev0/dev1
scores and exposure cannot be merged into this stage. Earlier three final sets
remain excluded. Source/voice/checkpoint roles remain fixed; named-corpus and
byte uniqueness do not certify physical-person/pretraining independence.

Use the unchanged calibrate_slr83_development.py 138-point grid, smoothing5,
event latch/refractory, zero negative-clip and pre-generation events, FAR95
<=0.5/h, and equal-voice-condition macro >=max(.48125, frozen v30 on this PCM).
Select maximum macro, then lower FAR upper, then higher threshold; no fallback,
extra exposure, rerendering or grid extension if none qualifies.

Compare frozen v30 (.91796875, smoothing5) and official Hi ESP (.63) on identical
full presentations and complete continuous background recordings. Preserve the
original official outputs, all Nyx scores and hashes. Report full-presentation
triggers and generation-reference delay; never call it true last-phoneme delay.
Report narrow inherited endpoint diagnostics separately from full-presentation
metrics. A development pass only enables a separately preregistered independent
acceptance stage, never automatic v30 replacement or human-generalization claim.

The runner is adapted from the unexecuted v36 reservation runner. All outputs
must be absent before execution; errors preserve partial outputs and stop,
without automatic retries or training restarts. Execution files and inputs are
pinned before launch. Source/license attribution remains in the original SLR83
source report and applies to all derived artifacts.
