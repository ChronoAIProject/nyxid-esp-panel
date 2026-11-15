# Exact training endpoints do not guarantee streaming support

Precommitted plan97b866ed059dda3b432ba506db82375c9b9d5556be909361681b7f99f1458fb3 completed all576 actualINT8calls, exit0. Same96 consumed training recordings, two frozen exports, exact source+fullRIR end offsets256/512/1024, same44800pastPCM frontend. No source clipping, FP32 recognition, threshold changes or new source consumption.

| Pooled stream miss | Codes at256 /512 /1024 | Earlier continuous waveform result |
|---|---|---|
|Piper428|221 /243 /191|rawpeak215, no event|
|Piper628|128 /62 /164|rawpeak194, no event|
|Qwen condition_011|253 /242 /248|rawpeak242, five-code peak1170<1175|

At original scalar threshold235/256, Qwen is above threshold at all three exact training endpoints. Piper428 crosses only at512, while no cadence-sampled stream call crosses. Piper628 remains below at all three. Exact endpoint calls are not480sample-spaced, so their values cannot be averaged as deployed five-step support. These observations isolate fixed-window position sensitivity from an assumption that training-source recognition itself is uniformly absent. They do not identify whether convolution padding, reflect-centered MFCC boundary phase, temporal averaging, or learned source cues cause that sensitivity, nor quantify quantization distortion.

The complete saved-batch audit establishes480000 fullphrase draws and fullsource+RIR containment for every draw. Piper perclass6048rendered records:192zero draws,3447at most3; median3,max11. SpeechT5 median319,Kokoro112,Qwenpositive92 (negative277.5). Equal engine mass is deliberately not equal rendered-record exposure. Native positive multiplicity and physical/pretraining identity limitations remain. Counts alone do not prove that oversampling Piper resolves the problem.

Inherited v38 already tried valid offset jitter0..140ms with preserved source mass on the old encoder; causal positivecoverage also traded recall for confusion without resolving Joe. A repeat of uniform jitter or positive share increases is not a new solution. A distinct future hypothesis would require explicitly measuring/controlling sensitivity across a physically continuous five-call trajectory, including all lexical negatives, alongside preserved source balance; architecture/frontend feasibility and previous five-score/temporal-risk failures must be addressed before locking a training run. Current evidence does not authorize another blind fit or guarantee such a loss works. Commands still lack direct training support. No qualification or P4 handoff; v30 unchanged.
