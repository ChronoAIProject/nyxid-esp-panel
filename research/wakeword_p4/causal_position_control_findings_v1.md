# Position-balanced complete-training control

The fixed5000step RTX4090 control finished in358.088seconds. All26initial tensors and all5000batch descriptors exactly match the original supervised baseline. Only the objective changed to0.5frame-normalized plus0.5position-normalized CE. No background objective, fresh development or sealed evaluation was used.

Preregistered complete4320training-wave replay: original530/720targets,149/3600negative recordings with an exact raw-argmax event; position-balanced520/720targets,39/3600negative events. Mix events fall84/720 to17/720 (bare20→1, light16→4, time15→3, timer33→9). Therefore short-position weighting substantially reduces consumed-training confusions, but slightly reduces target recall and leaves substantial errors. This is neither a qualified detector nor a demonstration of improved independent recognition or FAR. Do not promote or use sealed data to select another coefficient.

Both replay results retain per-record collapsed tokens and events, source hashes, and chunk37/full-logit agreement checks. Decoder is unchanged raw float argmax/exact ordered pattern without confidence/span/cooldown; results are deliberately not represented as integer-policy performance. Retain original baselines and the whole paired census. Frozen v30 remains unchanged.

Next investigate target-path insertions and remaining confusion by carrier/unit using this training census, and explicitly separate generator-duration alignment uncertainty from lexical discrimination. Any revised objective or temporal decoding assumption must be fixed before a new run; no hidden checkpoint/threshold rescue.
