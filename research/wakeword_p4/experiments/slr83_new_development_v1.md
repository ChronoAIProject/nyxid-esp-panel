# New development reservation, before any source generation or model scoring

Sources are the two fixed Piper OpenSLR83 checkpoints in
results/4090-piper-slr83-development-source-v1.json (source SHA
10f1c7b0351c6349fb200245b4ccb8a5cd68d929bd0759cfbc911838d51431d8).
Both are development-only; never move them to training or final. Model cards
and the primary https://www.openslr.org/83/ page identify the UK/Ireland
crowdsourced corpus and CC BY-SA4.0. Attribution: Google Inc., 2018/2019;
Demirsahin et al., LREC2020, Open-source Multi-speaker Corpora of the English
Accents in the British Isles. Retain source/license attribution with artifacts.
Different named corpus/checkpoints do not certify physical-person or pretraining
disjointness. Only two preset voices and one Piper architecture are represented.

Generate all two voices x fixed length multipliers .85/1/1.15 x five texts:
Hey Nicks., Hi E S P., Hey Nick., Hey Next., Hey Mix. Native config noise/length
settings unchanged except that multiplier. Save all native float output,
phonemes/token IDs/scales and resampled full PCM16, no trimming. ONNX gives no
phoneme durations or seed; preserve bytes rather than claim exact resynthesis.
Same mechanical gates: finite mono16k, .2..5s, RMS>=.001, no abs(sample)>=1 or
saturated PCM16. Retain failures; no score-based regeneration/filtering.
After the 30 clean sources pass, prepare fixed clean/25dB white noise/15dB white
noise plus synthetic RIR conditions; source RMS defines SNR, no RMS normalization,
retain full RIR tail and 1.8s lead/1s tail. Attenuate only to avoid clipping,
preserve original clean PCM exactly. Generation end and reverberant tail are
synthetic references, not last-phoneme truth. Do not crop to model context.

Before candidate scores, separately pin previously unused VoxConverse dev shards
2 and3 by index at the existing pinned mirror revision. This is a new candidate
selection stage, not an extension/rescue of the failed v30/v35 development gates.
Never combine their exposure with old dev0/1 to lower a previously reported FAR
bound. Audit full recordings, duplicate encoded/PCM hashes across old/new sources,
original exposure, continuity and provenance before scoring. All events count.
No final/test shard or VCTK reserved voice may be accessed at this stage.

Do not spend these new development scores until v36 completes its existing
exploratory regression and export audits. Use the fixed prior 138-point legacy
plus 1/256 threshold grid [.5,.99609375], smoothing5, original event latch and
refractory. Gate: background one-sided Poisson FAR95<=.5/h, zero negative-clip
activations and pre-generation positive events, equal-condition target trigger
macro >=max(.48125, frozen-v30 target trigger macro on these exact new sources).
Choose highest target macro, then lower FAR upper, then higher threshold. No
eligible threshold means null, no fallback/extra exposure/grid extension.
Compare official HiESP at frozen .63 and v30 at .91796875 on the same PCM;
report full synthetic presentation triggers and generation-reference delays.
Record that augmentation variants are correlated, not independent speakers.
These are development thresholds, not final acceptance. Any promotion still
requires a separately frozen protocol and untouched reserved final data, with
all original positive/negative/quantization/FAR gates intact.
