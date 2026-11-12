# Training-only US Piper coverage pilot v1

Motivation: v43 separates old training negatives at its historical threshold but
fails old development confusion/FAR. No further temporal sampler intervention
is currently justified. Add independently identified synthesis presets as a
coverage experiment, not old-development mining or a promise of better quality.

Use rhasspy/piper-voices immutable revision
`ae9d8352b73c1b1750748843e753a504b00e05b0`, en_US medium presets lessac,
ryan, ljspeech, one conditioning speaker each. Save original model cards,
ONNX/config bytes, origin URLs and SHA256. Cards describe Lessac and LJSpeech
as trained from scratch; Ryan is fine-tuned from rdh. Do not claim independent
physical-person/pretraining provenance merely from preset names. Amy was
considered from metadata but not selected because its card documents the same
Cori-Samuel base as existing British presets; no Amy audio generated/scored.
No VCTK or SLR83 preset, no existing development/final source is used here.

Before inference freeze this exact grid:3presets ×3 length multipliers(.9,1,1.1)
×4 exact texts(Hey Nicks.,Hey Nick.,Hey Next.,Hey Mix.) ×3 stochastic draws=108.
All are TRAINING-ONLY. Use original config inference noise and default
length_scale times multiplier, fixed tokenization via config espeak voice and
phoneme map. No pronunciation/threshold/voice/score search. Stochastic ONNX
output is preserved exactly; no claim of deterministic seed if not exposed.

Persist each complete native waveform, complete resampled16k float waveform,
PCM16 WAV, input phones/tokens/scales, generation order, model/config/card and
script/package identities, file+PCM hashes, sample lengths and whole-generation
reference end. Exposed token durations saved if available; waveform-only output
means durations unavailable, not inferred acoustic labels. No trimming, silence
removal, volume normalization, retries, failing-file replacement or score-based
selection. All attempts retained and all failures reported.

Mechanical gate per attempt:finite native and resampled values,mono/nonempty/
nonzero16kPCM16,.2..5s,RMS>=.001,no abs(native)>=1,no abs(resampled)>=1,
no -32768/+32767 serialized samples, exact native→resampled→PCM roundtrip under
pinned converter. This is process compliance, not phonetic truth/listening
quality. Hash audit separate from generation. Failures cannot be hidden to claim
all108 passed. No wake-model scoring or automatic training/promotion in this
pilot. Any later dataset/augmentation/sampler requires separate frozen plan.
Old-source40endpoint failures and historical native-float gaps remain. New
untouched development/final and frozenv30 remain unchanged.
