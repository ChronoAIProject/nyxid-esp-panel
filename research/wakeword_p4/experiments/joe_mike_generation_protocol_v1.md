# Joe/Mike whole-output source reservation v1

Reserve Joe for development and Mike for independent acceptance, before any
new audio generation or detector scores. Neither is training material. Keep
all existing training sources and the three original final sets unchanged.
The two voices have distinct model/config bytes and declared target voices,
but share the rdh parent. This is a voice/model holdout, not parent-family or
certified physical-person/pretraining independence. Known cross-machine
consumption checks found no prior use; missing records are not proof.

The machine-readable generation plan pins every requested text, phone/token
input, model/config, runtime, speed and replicate before synthesis. Four
Hey Nicks target phrases and four corresponding Hi E S P phrases include
standalone and command contexts; twelve common confusions cover near words
and partial phrases. Every voice has all twenty texts at length scales
0.9/1.0/1.1, two draws each:120 native outputs per voice. Same text/speed/draw
indices pair the two intended wakephrases, not identical speech PCM. Both
detectors must later consume every identical rendered PCM, including
cross-target negatives. No post-score text/voice/speed/draw selection.

Use pinned ONNX directly on CPU, two threads, sequential execution, noise
scale0.5/noise_w0.333. RandomNormalLike exposes no per-call seed or durations;
save complete actual waveform bytes, available input phones/tokens and null
duration metadata. No promise of bit-identical re-synthesis. Native output
is saved immediately as float32 NPY; complete rational resample to mono16k
PCM16, no gain/crop/normalization. Apply unchanged mechanical gate v1 to every
requested row. All failures/errors preserved; any failed or missing row
prevents pack qualification. No repair or replacement draws under this plan.

This reservation authorizes only Joe DEVELOPMENT native synthesis and its
mechanical audit. Mike remains unopened at inference until the complete
rendering/model/threshold/acceptance protocol is committed and development
passes. No wake-model scoring is authorized by this component. A native
mechanical pass is not pronunciation truth, a trained candidate or promotion.
No human listening/recording requirement. Whole generated output end will
be a declared synthetic reference; for commands it is not wakeword end.

Before scoring: freeze whole-tail RIR/noise/render and padding records,
candidate identity and original checkpoint selection, official Hi ESP and
frozen v30 binaries/runtimes, streaming/reset/quantization contract, finite
threshold grid/ties, positive/confusion gates and FAR one-sided95<=0.5/h.
Use reserved test4/5 whole-waveform development (7.9028622222h), test6/7
acceptance (unopened, at least6h); no original final sets or consumed sources
for threshold selection. Preserve all development failures. Acceptance only
once after locking operating points; no rescue with another checkpoint,
new threshold, more duration or regenerated positives after seeing results.
Training Complete-recovery control/recovered jobs remain untouched; this
source reservation does not authorize inherited-positive formal training.
