# Training allocation / exact decoder compatibility

The metadata-only oracle over all3456 prospective training records gives576/576 target recordings activated and0/2880 negative recordings activated. It takes the declared RIR-weighted overlap soft labels, maps to the fixed dense training vocabulary using the original zero-fill rule, and sends argmax tokens to the unchanged raw exact matcher. No waveform, trained model, fresh development or protected final data is read.

This rules out the narrow explanation that these training allocation labels necessarily lose a required unit under argmax or necessarily produce false exact paths. It does not prove acoustic boundary correctness, a bound on model recall, confidence-policy compatibility, INT8 viability or generalization. Do not change labels or relax the decoder solely on the rejected candidates' aggregate errors. A next alignment hypothesis requires stronger evidence than mandatory loss of units by this allocation geometry.

The first audit invocation failed before writing an artifact because a per-record target matrix can have fewer columns than the global vocabulary. Corrected construction matches `causal_unit_full_batch_v1.py`: zero-fill absent columns and assert retained probability mass1. There was no training or model invocation in either attempt.

Script: `experiments/audit_generator_label_decoder_v1.py`; original complete metadata results and input hashes: `results/4090-generator-label-decoder-compatibility-v1.json.gz`.
