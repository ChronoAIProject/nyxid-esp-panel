# Shorter-span daily and background precision identities

The treatment removes the one observed daily common-name conflict, but it does not remove the observed full-background risk: the single Melo event moves to a different recording. Both changes already occur in float, with identical float/INT8 event steps for the implicated recordings. Counts alone hid this identity change.

- Complete96daily clips: control float/INT8 both1event, index75 “Hey Nick, what time is it?”, step198 /13920relative samples. Treatment float/INT8 both0events.
- Complete12Legacy background streams, reused10.483944826h: both arms and precisions0events.
- Complete12Jenny background streams, reused9.614419583h: both arms and precisions0events.
- Complete42Melo background streams, reused7.902862222h: control float/INT8 event on ppexo.wav, index447, step4311 /1988160relative samples (124.26s). Treatment float/INT8 event on ofbxh.wav, index471, step26822 /12793440relative samples (799.59s). Each arm has1event, but the recording and time differ.

See4090-shorter-span-float-interference-census-v1.json for exact source and trace identities bound to the complete reviewed float reports. These are consumed synthetic/dataset diagnostics, not daily incidence or new independent background exposure. Do not pool reused hours. No additional inference or tuning was performed.

Removing the Nick sample is useful under the user's priorities, but does not compensate for large normal crossvoice losses. Keep C161 baseline and existing dual/triple handoffs; no shortspan promotion. The neural-span change alters float target and non-target decisions, rather than merely removing old history or repairing quantization. Any next intervention must preserve normal support and address ordinary-speech risk without a span/seed/threshold sweep or reopening failed teacher/consistency recipes.
