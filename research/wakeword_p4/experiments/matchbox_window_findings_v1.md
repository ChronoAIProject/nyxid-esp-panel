# Pretrained-window v1: training streams fit, consumed Melo recall fails

The original fixed5000update/seed726 RTX4090 experiment completed in115.75s;
all finite-loss/gradient, optimizer-step and frozen-BN-statistics checks passed.
One original builtinINT8 export completed:115408bytes, SHA256
`af644f35a9aa3eb65293f5ee780dcab5d95e842d0be1d63a3d6d9307f8052988`.
No checkpoint, learning-rate, context, threshold, head or calibration rescue was
performed. This candidate does not replace frozenv30 or the Mac c161 trial.

This is a different frontend/encoder/task package: pinned original MatchboxNet
weights, natural-log64MFCC, symmetric convolutions/global pooling, a new7class
head, and74247trainable parameters. All training sources were already designated
training-only:378SpeechT5 bare presentations plus1004Libri original negative
utterances. No Melo training or protected/acceptance audio was used. This is not
an isolated causal estimate of pretraining or a repeat of the old scalar recipe.

| Actual deployed INT8 diagnostic, fixed0.91796875 | Target hits | Commands | Common confusion | Cross target |
|---|---:|---:|---:|---:|
| Complete SpeechT5 training bare waveforms |63/63|not present|0/252|0/63|
| Complete consumed Melo waveforms |2/72|2/54|0/288|0/72|

Training runs include all7speakers,3speeds and3conditions. Melo bare is0/18
across both conditioning IDs. All474Melo/background streams completed;
0events/7.902862222222223h gives the descriptive one-sided95%Poisson bound
0.3790692776004911/h. This reused consumed background is not new independent
exposure or a qualified operating point. Training-only378waveforms have no
background exposure; their FAR is null, not zero or an infinite-confidence claim.

Both waveform groups used the identical exported model, frontend buffers,
20480sample past-PCM context,480sample cadence, continuous81600zeroPCMprime,
five-UINT8 arithmetic, half-threshold rearm and67step refractory. The narrowly
adapted training collector changes only metadata/population/report/FAR fields.
All source/trace/input/runtime hashes, prime and finish-call events remain.
This verifies that the large training/consumed-bare difference is also present
under the same actual wake-event function, not just crop classification versus
streaming metrics or absent command training. Voice/engine/duration/noise/carrier,
window distribution and quantization coverage remain confounded; no single
cause is isolated or source quality/phonetic correctness inferred.

The separate129training/calibration-window check reports16/16Nyx codes255 and
all112otherclass codes0; literalzeroPCM also yields0. It contains no FP32 model
outputs or CPU/GPU neural agreement test. These already fitted/calibrated windows
are not heldout samples or appwake metrics. They show basic exported-model
classification capability, not absence of every possible quantization issue.

Consumed Nyx post-source five-score-peak median is0.018359375, p750.3548828125,
maximum0.97421875; Next reaches0.91328125. These distributions do not license a
new threshold or demonstrate a threshold-only solution.15/72Nyx generated-prefix
spans exceed1.28s; many shorter spans also miss. Prefix spans include internal
synthesis durations/padding, so this is a context-capacity observation, not proof
that audible phonemes were cut or a causal explanation of all misses. No sources
were trimmed, filtered, regenerated or reassigned after seeing scores.

Melo's two first-hit generated-prefix signed times are8.0625/73.6875ms; training
first-hit full-bare-container median is-64.5ms. Populations and generation clock
definitions differ; these do not establish a latency ranking or human last-phone
timing. This is offline CUDA float32MFCC plus CPU builtin/defaultdelegate INT8
replay, with logical support/return clocks, not measured P4 realtime behavior.
No state or readiness censoring occurs at prime/source transition. Bounded
window reflection uses available pastPCM only. Final partial block's virtualzero
padding and actual clamped clock are marked explicitly.

Model conversion estimates9.279M MAC per30ms call, about309M MAC/s excluding
frontend/overhead, substantially more than the existing incremental model.
The external MFCC frontend still needs a P4 implementation; flash size alone
does not establish performance or whole-device resources. No board handoff or
reliability claim follows from this failed consumed diagnostic.

The next hypothesis must address evidence from both actual waveform groups and
inherit older failed interventions. Do not blindly extend budget, change window
or lower threshold on these consumed clips. Additional source/task-support work
would need its own concrete factor, complete source-quality/role audit and
prospective lock; future promotion still needs new unconsumed development and
the sealed independent acceptance. Full research goal remains incomplete.
