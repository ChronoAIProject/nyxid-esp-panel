# Prospective causal candidate protocol

The current causal controls consumed all20 conditioning voices/variants, so their reports are training diagnostics only. The new prospective split reserves `af_alloy`, `af_kore`, `am_echo`, and `am_santa` (864 complete records: 4 conditioning × 4 carriers × 6 classes × 3 variants × 3 speeds) for development. The remaining16 conditioning are the only optimization pool for a new candidate.

A new run must construct its sampler from the split manifest, verify every PCM/file hash and role, and fail if a development conditioning appears in a training batch. It must use the fixed causal frontend/cache ABI, complete recordings, no future PCM, and a single predeclared objective/decoder. The development partition is opened only after the final checkpoint and real-state INT8 export are immutable. One policy/threshold is locked there, with target hits, confusable hits, complete-background events, event timing and one-sided Poisson upper bound recorded. No protected final set is opened until this lock is written.

The split is synthetic and conditioning-disjoint, but still shares the TTS/rendering family and may share unknown source/pretraining lineage. Results are limited to this protocol and do not establish human pronunciation, human generalization, or Hi ESP equivalence. Frozen v30 remains the handoff baseline until every promotion gate passes.
