# Binary40 background event: ordinary-speech context evidence

Prelocked65a6e940 single whole-recording audit completed actualexit0, session63962. Existing pinned Whisper small.en CUDAFP16, beam3, temperature0, English, no prompt/VAD/previous-text conditioning, CPUthreads2. Entire178.432s original ppexo.wav transcribed once;3.122449477s ASR elapsed. All segments/word timestamp estimates preserved in results/4090-binary40-background-context-original-v1.json, original execution output preserved losslessly. No crop audio, new synthesis, wake reinference, model fitting, relabeling or parameter changes.

Original source is already-consumed development shard test-00004-of-00011.parquet, row15, bound to its existing diagnostic SHA. Metadata contains speaker intervals but no transcript. Event124.26s is inside the original117.2–128.44s speech interval. Last source speech annotation ends137.13s while full audio lasts178.432s; do not trim the tail or treat timing validity as complete transcription coverage.

Whole-file ASR context:

- 111.22–116.04s: 'whose loss is no less profound and our hearts go out to their friends and families too,'
- 116.82–124.70s: 'I wish I could say something to make sense of it all and I can’t find the words to say.'
- 124.76–132.06s: 'All I can think of is this, if you can, take a moment, tonight, tomorrow, to call up someone'
- 132.06–134.56s: 'you love and just let them know.'

Estimated words around the event: 'I'121.56–121.86, 'can’t'121.86–122.82, 'find'122.82–123.22, 'the'123.22–123.40, 'words'123.40–123.70, 'to'123.70–124.50, 'say'124.50–124.70. These are ASR estimates, not phone alignment or independent human listening. The detector's block-return event is not the exact sound or word causing activation; a306-frame receptive field plus smoothing uses prior context. Do not attribute it solely to 'to' or 'say', or infer a label defect.

Saved raw traces independently authenticated, without new recognition: at eventstep4311 Binary40's last five codes are[243,243,248,253,243], sum1230 versus fixed1175 threshold. Original C161 at the same step has[27,20,20,20,13], sum100; its maximum single code anywhere in this complete recording is27. The event is a sustained five-call crossing, not merely a lone spike. Evidence is results/4090-binary40-background-context-review-v1.json. No threshold sweep or alternate cropping was performed.

Interpretation: current evidence supports an ordinary-speech false-activation concern, rather than automatically dismissing this as a rare constructed nearword. ASR did not render a wake phrase in the displayed context, but transcription cannot prove absence or phonetic truth. Retain the original background event and its full exposure unchanged; do not relabel it positive, remove it, or admit this development recording into training. It remains one event in the separately reported reused7.902862222h cohort, not a new exposure period or a measured population frequency of this sentence.

Decision unchanged: C161 remains baseline and Binary40 remains a promising research candidate. Its continuous/Joe gains coexist with ordinary bare/command regressions and this everyday-speech concern. This one audit does not justify abandoning the architecture or sweeping its thresholds. Next evidence should distinguish representation/convergence and quantized behavior across the complete already-declared training diagnostic, without optimizing on this development excerpt. Mac/P4 numerical/retention/resource/stability evidence remains independently incomplete.
