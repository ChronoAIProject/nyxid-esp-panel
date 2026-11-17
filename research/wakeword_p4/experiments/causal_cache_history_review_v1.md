# Explicit causal cache: history correction and decision boundary

2026-11-17. The early feasibility note accurately describes its then-untrained export. It is not a statement of the current research history. The prior goal refresh mistakenly described the architecture as untrained; this review corrects that before any training.

Current training/causal_cached_encoder_v1.py SHA256 is 5bbcb40be78dd3b32e9124c9380665d764379d83e16e6db977c61cfe7c2058f1. Five original training reports bind exactly this hash: unit, position, delay40, conditioning-uniform and conditioning-risk. Their report hashes are recorded in results/4090-causal-cache-history-audit-v1.json. This is a scoped search of *training*v1.json provenance maps, not an exhaustive claim that only five runs used it.

The generator-unit model directly instantiates this encoder and retains a per-frame Conv1d readout. Historical findings at the research root establish:

- causal_unit_supervised_findings_v1.md: 5000 RTX4090 steps, all4320 training recordings sampled, first24 training diagnostic target2/4 and negatives2/20. Full/chunk and trained Keras transfer checks passed; these do not establish detection quality.
- causal_prospective_development_findings_v1.md: conditioning-disjoint contextual candidate target47/144 and negatives29/720 on development. No successful transfer claim or INT8 promotion.
- causal_conditioning_risk_findings_v1.md: balanced control development54/144 versus risk51/144; negative27/720 versus21/720. Risk rejected for target losses.
- contextual_allocation_margin_findings_v1.md: weighted development46/144 versus control54/144, negatives30/720 versus27/720; rejected.

These use generator-allocation supervision and raw exact-sequence matching, not C161's binary score and five-call event policy. They cannot be numerically compared as a C161 benchmark, and they do not prove every possible binary head on this encoder fails. Conversely, merely retraining the same retained-unit readout is not a new architecture experiment.

A potentially distinct intervention is a binary readout with the original broad C161 source-balanced supervision, preserving source schedules and masks. It is not yet selected: the existing encoder takes64 MFCC features every10ms, while the actual C161 training feature shape, normalization, frame clock, effective temporal support and deployable frontend must first be reconciled. Changing the frontend as well as architecture would be a combined intervention and must be disclosed. Do not imply the old analytic INT8 calibration qualifies a trained detector; actual stream-state calibration, quantized cache feedback, complete binary graph export and P4 resource evidence remain necessary.

No new optimization, synthesis or replay was launched for this review. It avoids a historical repeat and narrows the next action to a concrete input/temporal-support compatibility audit, without inventing a requirement for another new loss. C161 remains the development baseline; all prior models/reports and the two unrelated untracked original diagnostic JSONs remain untouched.
