# Causal MFCC style augmentation control v1

The training-only affine MFCC perturbation control was trained for the fixed 5000 steps on the 16-name conditioning-disjoint training partition. Gain was sampled from [0.85, 1.15], bias from [-0.15, 0.15], seeded by plan seed plus step; inference graph was unchanged.

Complete-wave raw-float exact ordered matcher replay:

- training partition: 437/576 target recordings, 39/2880 negative recordings
- reused development partition: 65/144 target recordings, 39/720 negative recordings

The development result fails the preregistered synthetic acceptance gates. This control is rejected, no INT8 export is made, and frozen v30 remains the only approved Mac handoff. The development material is reused diagnostic evidence and is not fresh independent acceptance evidence. Results are synthetic-protocol findings only.
