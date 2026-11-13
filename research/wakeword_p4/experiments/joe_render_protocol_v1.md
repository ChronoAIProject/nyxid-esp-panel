# Joe development presentation reservation v1

Render all120 qualified native-source PCM files, each under exactly three
conditions: quiet, seeded25dB white noise, seeded15dB white noise plus fixed
finite synthetic room response. No detector inference is authorized here.
No row selection, gain search, source trimming or regeneration. Keep original
native tensors/PCM and original generator rank-error records unchanged.

Use the existing independently exercised synthetic_render_quality_v1 contract:
mono16kPCM16, 28800sample quiet lead and16000sample tail; RIR impulses at
0/160/480/960 samples with amplitudes1/.25/-.12/.06. Keep full convolution,
including all960tail samples. Seed derives from SHA256(source requestID plus
":joe-render-v1:" plus variant), first16hex as integer, fixed before render.
Noise spans the complete padded presentation, RMS set from complete dry
source RMS and requestedSNR. Quiet gain1; noisy variants deterministically
apply min(1,.95/peak-after-mix). This is prescribed presentation gain, not
repair of native synthesis or a score-adaptive gain. Record every gain and
all seed/SNR/source/RIR boundaries. No gate relaxation on failure.

Renderer uses FFT convolution; validator reconstructs direct convolution and
every mixed sample independently, checking complete native/RIR references,
length, quantization, finite/no-clipping, no saturatedPCM and padding. Every
one of360requested renders must pass; missing/error/failing rows remain and
prevent qualification. Mechanical gates do not certify phrase semantics,
phonetic truth, physical speaker independence or detector performance.

Legacy validator field keyword_end means whole-source-plus-RIR end here;
also record explicitly generated_output_end_sample and full_rir_end_sample.
Commands use complete generated container end, not an invented wake-prefix
or human phoneme endpoint. Later streaming analysis must report early/prefix
events and negative as well as positive signed end-reference latencies; never
drop early events to make latency look better. Report standalone/command
separately. Same PCM must feed official HiESP/v30/candidate engines.

Mike acceptance is not generated or rendered by this component. Roles stay
Joe development/Mike acceptance, neither training, with sharedrdh correlation
disclosed. Acceptance rendering policy must be pinned before Mike inference;
no content-dependent retries after scoring. Full candidate/binary/threshold/
acceptance protocol is still required before ANY new wake-model scores.
