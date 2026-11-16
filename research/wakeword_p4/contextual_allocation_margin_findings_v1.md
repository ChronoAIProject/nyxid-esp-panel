# Allocation weighted contextual margin: rejected

The fixed allocation weighted contextual margin training completed 5000 updates on RTX4090 in 595.658 seconds, with finite gradients. The only objective change was replacing the uniform mean over selected nonzero support frames with normalized declared allocation mass. Initialization, sampler schedule, frame/position losses, coefficient0.1, optimizer, decoder and budget were fixed. The functional objective audit passed uniform equivalence, scale invariance, zero-mass behavior and invalid-weight rejection.

Complete raw-float exact matcher replays:

| Arm | Training targets | Training negatives | Reused development targets | Reused development negatives |
| --- | ---: | ---: | ---: | ---: |
| Uniform balanced contextual (completed control) | 439/576 | 46/2880 | 54/144 | 27/720 |
| Allocation weighted contextual | 425/576 | 36/2880 | 46/144 | 30/720 |

The weighted candidate loses8 development target hits and adds3 negative hits. It is rejected before INT8 export, threshold selection, or any handoff. The training provenance audit confirms identical initial model and all5000 batch descriptors; no checkpoint or coefficient rescue was used. Development evidence is reused diagnostics, not fresh independent acceptance. Counts are raw-float recording activations, not confidence-policy or FAR qualification.

This result rejects the fixed weighting intervention, not all possible allocation-aware objectives. It does not establish acoustic or phonetic truth, human generalization, or Hi ESP equivalence. Frozen v30 remains the only approved model.
