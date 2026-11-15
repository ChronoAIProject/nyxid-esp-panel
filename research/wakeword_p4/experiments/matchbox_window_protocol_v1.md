# Fixed pretrained-window experiment v1

This is a new architecture/frontend/task package, not an isolated causal test of
pretraining or a rescue of the closed Melo v2 cohort. No Melo source enters
training. Original failed CTC, auxiliary heads, six-class competition, scalar
loss and source-coverage results remain. The current hypothesis is that a
publicly pretrained small command encoder plus explicit matched phrase
classification may distinguish synthetic nearwords across sources better than
the existing causal scalar representation. Basic GPU function is proven;
recognition improvement is not.

Deployment behavior is prospectively finite-waveform-window inference: retain
the latest20480PCM16samples, initially zeros; process every480new samples. Each
call computes its own centered512FFT/400Hann/160hop log-MFCC from stored original
mel/DCT buffers, takes first128of129frames, and runs the original symmetric
convolutions/global mean pooling with a new7-class softmax Nyx output. Reflection
is relative to each bounded available window, not to future recording samples.
This is a causal app function of available past PCM, with noncausal operations
inside that bounded window; it is neither a causal-convolution conversion nor
the historical continuous MicroFrontend. Explicit boundary effects and full
recomputation cost remain. No future waveform or known evaluation source end
is used to gate notifications.

Preserve the same81600zeroPCMprime and480sample app cadence, decoder state and
all priming events. Final partial block pads zeros to480 once; preserve nominal
support and actual clamped source clock and mark finish-call events. Never reset
the raw context, MFCC convention or decoder at the prime/source transition.
Future diagnostics use original five-UINT8 arithmetic, half-threshold rearm and
67step refractory, fixed0.91796875 without threshold rescue. Same full PCM as
v30/reference is required; finite-window and native block differences are
reported. This protocol itself does not authorize new fresh-data scoring or
replace the closed development selection policy.

Training material is the exact pushed1382source manifest:378SpeechT5 full bare
presentations and1004originalLibri training utterances from251previously audited
training speakers. Three positive/full-nearword placements retain every
transformed source and complete RIR tail; after-source padding is256/512/1024.
Each128batch draws16matched same-voice/speed/condition groups, all6phrases each
(96inputs), plus24uniform training utterance windows and8windows wholly before
source onset from original synthetic leading ambient. Speech has no endpoint
or command labels invented. Short background utterances pad zeros. PCM16 is
scaled by32768; no new augmentation, TTS, endpoint, human listening or source
selection is introduced. Whole original raw files/decoded PCM hashes remain.
Natural negative transcript screening is not a phonetic guarantee; pretrained
physical-speaker overlap is unknown.

Strictly load pinned original35class weights, then replace only final128→35
linear layer by128→7: background/Nyx/HiESP/Nick/Next/Mix/Muse. Seed726, all74247
parameters trainable, model.eval keeps9BNrunning-statistics sets fixed; BN affine
parameters are trained, dropout0. Original MFCC buffers do not train. Plain
unweighted categorical cross entropy, freshAdam lr1e-4/betas(.9,.999)/eps1e-8,
weight_decay0/foreachFalse/fusedFalse; constant5000updates, batch128 onRTX4090,
FP32/TF32off, no scheduler, early stop, checkpoint/seed/LR/head search or rescue.
Save initial/final weights, optimizer, exact batch descriptors and representative
bytes. Loss/gradients must be finite; frozen-buffer integrity is an invariant,
not a CPU/GPU model-numerical equivalence admission.

Before first update, save the actual first batch of128MFCC windows and one
literal zero-PCM MFCC window as fixed129representative inputs. One final
builtinINT8 export only, standard depthwise/pointwise/residual/ReLU/globalaverage/
softmax/slice operators, input[1,128,1,64]INT8 and Nyx[1,1]UINT8. No calibration
retry, FP32 numerical-agreement gate or intermediate checkpoint choice. Export
implementation/ABI lock must be pushed before conversion. Repeated-window CPU
deployment preprocessing is separate from GPU training; evaluate actual exported
bytes on complete waveform streams before any conclusion.

This is training/consumed diagnostic feasibility only. No promotion or new
threshold selection, no open protectedMike/historicalfinal/EN-INDIA/EN-AU audio.
Any future qualification needs a separate prospective unconsumed-development
population and an unchanged sealed independent acceptance protocol; consumed
Melo scores cannot become fresh evidence. Frozenv30 and Mac c161 remain.
