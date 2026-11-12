# Balanced inherited-training conditioning-ID synthesis pilot v1

Evidence after rejected v45: old training negatives have no historical high
peaks, yet31old-development negative failures occur in actualfloat and INT8.
Three newUS presets fit well but do not establish broad identity separation.
Authenticated inherited Piper aliases cover644/672 allowed conditioning IDs
for positives, only216/217/208 for Nick/Next/Mix. Pair membership overstates
single-ID acoustic coverage; it is not proof of independent physical people.
Existing British/source-mass/time/loss/capacity experiments did not pass gates.

Hypothesis for later testing: balanced matched positive/confusable synthesis
across the existing training conditioning IDs may improve separation beyond
three single-speaker US presets. This is source/phrase joint coverage, not more
US mass or another endpoint experiment. Old quality gaps additionally motivate
new complete native outputs, not repairing/relabeling old audio. No guarantee
of generalization; shared LibriTTS pretraining remains documented.

First a process-only pilot: reconstruct original training partition with
numpy default_rng(726).permutation(904)[:672]; use its FIRST32 IDs in original
order, frozen in 4090-piper-training-phrase-coverage-v1.json. Never select IDs
from failures. All remaining IDs remain training-only, not unseen evaluation.
No reserved conditioning ID, waveform, replay manifest, feature or score read.

Use existing kahrendt/piper-sample-generator revision
b5b36e266576693ea47db129fbfe6bbc06164762, original model
.local/wakeword/piper/models/en_US-libritts_r-medium.pt SHA256
e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6,
config SHA256119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848.
Record actual local generator/helpers/package/source hashes, including local
import-only modifications. Source URL metadata does not prove byte identity
without fetched hash comparison. Model is trusted existing research input.

Fixed grid32IDs ×3length scales(.9,1,1.1) ×4texts(Hey Nicks.,Hey Nick.,
Hey Next.,Hey Mix.) ×1draw=384 attempts. Same speaker twice (no interpolation
between different IDs),slerp.5;noise_scale/noise_scale_w from config (.333 each).
Torch seed187726, one-example generation in fixed ID/speed/phrase order.
CUDA4090, no torch/TF co-resident process. Exact CUDA reproducibility not assumed.
Save exposed token durations and input phones/tokens; these are generation
process references, not acoustic truth. Save complete native float32 and
resample_poly320/441 float32,PCM16WAV, all source/run/package hashes and order.
No normalization/trimming/retry/failing replacement or wake-model scoring.

Use unchanged mechanical gates from US pilot: finite,mono/nonempty/nonzero,
.2..5s,RMS>=.001,no native/resampled abs>=1,no PCM endpoints,complete length,
exact pinned-library roundtrip. Every failure retained and counted. Save all
384attempts even if gate fails; no posthoc tolerance or audio repair. Verify
all duration arrays, full lengths, file/PCM hashes and exact conversion using
independent read-only audit before any feature pack. Shared-library limitations
explicit. New corpus cannot retroactively certify old native-quality provenance.

No formal v46, training replacement/mass/augmentation/checkpoint change,
threshold selection, development/final scoring or promotion in this pilot.
Only after full pilot evidence may a separately frozen larger balanced grid
and training design be considered. V30 stays frozen; all original acceptance
and fixed old-stage gates remain unchanged. No human recording or board needed.
