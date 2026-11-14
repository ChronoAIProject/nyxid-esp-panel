# Next feature and training stages

The original 7740 native requests and 23220 complete renders have completed
process-quality and independent saved-array/replay checks. Narrow and broad
each reference 12960 presentations: 2592 positive, 10368 negative. Shared
sources/environments are referenced by both arms, never regenerated.

Extract the unchanged MicroFrontend from each whole serialized presentation
once into a shared uint16 binary. Build separate narrow/broad metadata views
with explicit shared offsets and exact source-role/label hashes. Hard-link
the shared binary and authenticated original training-only human/ambient
arrays; never copy validation/testing arrays into either arm. Preserve every
frontend frame; no crop or trim at extraction. Independently verify complete
binary/offset/source/label coverage and preselected frontend replay witnesses.
Check the disk budget before extraction.

Both arms use the same geometry-only context309 and lead49920 decision,
computed from all complete training positives without detector responses.
The boundary for command positives is the whole generated command output,
not an acoustic wake-prefix endpoint. ONNX token durations are unavailable.
Consequently **prefix_negatives=0**: an arbitrary crop before a command's end
may already contain the complete wakephrase and cannot be assigned a negative
label by pretending that command-end is wake-end. Use the explicitly generated
partial/near-word and cross-target negatives instead. This applies to both
arms; no historical run/configuration is amended.

Prospective matched training: broad is the sole candidate; narrow is a
diagnostic control, never a fallback candidate if broad fails. Single fresh
20000-update run per arm, fixed final20000 only; snapshots are evidence, not
checkpoint selection. Same pretrained teacher60 convolution/BN tensors;
initialize the context309 classifier once, preserve the actual full arrays
before either run, and use the identical initialization for both arms.
Freeze BN. Original Adam1e-5/beta1 .9/beta2 .999/epsilon1e-7, no extra head or
domain loss, original BCE and batch128 (32 positive,48 synthetic negative,
48 original training ambient; weights1/20). Equal TTS model probability,
seed726, no time/frequency mask or hard-negative mining. Whole-positive
completion jitter3..27; complete negative timeline sampling. Any required
option or numeric detail must be serialized in a pushed execution lock
before fitting; this document alone does not launch training.

Before formal training, require actual CPU batch/label/role/whole-source
checks, preserved initialization/zero-optimizer state, both two-step GPU
integration smokes and actual streaming export contract checks. Do not
reintroduce cross-CPU/GPU gradient/logit equality experiments. A smoke is
integration evidence, not a substitute for the full run or recognition.

After fixed-final export, qualify actual waveform-streaming INT8 mechanics
and preserve complete scores on consumed regression separately from fresh
development. No LR, coefficient, seed, speaker, checkpoint or threshold-grid
sweep to rescue failure. Independent multi-voice development/acceptance,
frozen-v30 and official-HiESP same-PCM replay, unchanged preregistered
positive/negative and FAR upper95<=.5/h gates remain outstanding. Jenny,
Mike, paired acceptance and old finals retain their protected roles. No
human listening or unknown physical/pretraining certification prerequisite;
state the actual documented separation and its limits.
