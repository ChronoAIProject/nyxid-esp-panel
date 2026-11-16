# MFCC channel masking robustness control: rejected

The training-only channel mask control completed5000 fixed CUDA updates on RTX4090 in546.412 seconds. Each training batch independently masked each of64 MFCC channels with probability0.125 using seed726+step; inference graph and all labels/losses remained unchanged. Admission checks and finite-gradient checks passed.

Complete raw-float exact matcher replay:

- training partition: target246/576, negative24/2880
- reused development partition: target28/144, negative18/720

The candidate loses substantial target coverage and is rejected before INT8 export, threshold selection or handoff. Its lower negative activation does not compensate for the target regression. No checkpoint, mask probability or threshold rescue was used. The development set is reused diagnostic material, not fresh acceptance evidence. Results are synthetic protocol diagnostics only; no human generalization, phonetic truth or Hi ESP equivalence is claimed. Frozen v30 remains unchanged.
