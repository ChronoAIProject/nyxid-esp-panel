# Causal MFCC style augmentation control v1

The training-only affine MFCC perturbation control was trained for the fixed 5000 steps on the 16-name conditioning-disjoint training partition. Gain was sampled from [0.85, 1.15], bias from [-0.15, 0.15], seeded by plan seed plus step; inference graph was unchanged.

Complete-wave raw-float exact ordered matcher replay:

- training partition: 437/576 target recordings, 39/2880 negative recordings
- reused development partition: 65/144 target recordings, 39/720 negative recordings

The development result fails the preregistered synthetic acceptance gates. This control is rejected, no INT8 export is made, and frozen v30 remains the only approved Mac handoff. The development material is reused diagnostic evidence and is not fresh independent acceptance evidence. Results are synthetic-protocol findings only.

## Saved-result stratification and replay provenance

The exact replay script formerly retained in `/tmp/replay_style.py` is now tracked as `experiments/replay_styleaug_generator_v1.py`, preserving SHA256 `a6974cbfcd4deb05becab48fbd44ee206b39235d94810cbcea219eb9946f71a8` recorded in both original reports. No model replay was repeated to recover provenance.

Saved development results stratified by conditioning show styleaug target hits 19/36 alloy, 21/36 kore, 17/36 echo and 8/36 santa, with negative hits 11/180,19/180,7/180,2/180 respectively. The normalized candidate had 15,20,16,0 target hits and 1,16,2,0 negative hits. Styleaug false clips are Nick6, Next16 and Mix17 (each /144); normalized has Nick7,Next11,Mix1. This comparison changes architecture as well as training augmentation and cannot isolate the causal effect of augmentation. Broad conditioning failure and increased confusable activation remain; these diagnostics do not justify merely increasing the same affine augmentation. Machine-readable counts and input hashes are in `4090-styleaug-development-strata-v1.json`. No fresh data, acceptance data, thresholds or model invocations were used.
