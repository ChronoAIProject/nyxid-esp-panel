# Multiscale float interference identities

The newly observed US whole-context Max event is identical across float/INT8: step6786, relative3176160samples. Control has no event in this container in either precision. Thus the named common-speech conflict predates export quantization.

Both complete96daily sets retain the same Nick clip75. Control float/INT8 step198/13920samples; treatment bothstep204/16800samples. Same source conflict,180ms later. Melo background in both precisions remains the same ppexo.wav index447: control4311/1988160; treatment4307/1986240. No background-recording swap in this experiment.

Additional treatment float-only background events, absent from original INT8 and from control float:

- primed: source4423872, index10, events{'float': [{'step': 96042, 'relative_sample': 46019040}], 'int8': []}.
- jenny: source4483046, index7, events{'float': [{'step': 1355, 'relative_sample': 569280}], 'int8': []}.

Complete reused exposure is Legacy10.483944826h, Jenny9.614419583h, Melo7.902862222h, perarm/perprecision; these are not new independent hours and must not be pooled. Background risk cannot be summarized by target-hit improvements or assumed monotonic quantization harm.

See audit_multiscale_float_interference_v1.py and4090-multiscale-float-interference-census-v1.json: full daily/background/context saved-event census, original source descriptors, trace identities and bound report SHA256. No new inference/training/tuning. Prior complete review authenticated saved arrays/events. Container attribution does not establish phonetic cause or daily incidence. C161 remains baseline, no model promotion. Latest Issue18 checked; no newer Mac engineering reply yet.
