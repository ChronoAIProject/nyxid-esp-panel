# Independent original-probe loss reduction

The original eight NPZs were delivered at GPU `3f656a2` as
`gpu-actual-raw-consistency-saturation-v1.tar.gz`, 4259794 bytes, SHA256
`57e4c3b2af9f1b9c0e0d4b081a37859320a3294bbd9a04bb61cd5a056ee78ba7`.
They are initialization training probes, not final models or recognition data.

The native Mac reader authenticates the archive, manifest and all 13 payloads,
including the five published metadata/source files against their original Git
receipts. It does not execute the packaged producer/reducer. It hashes each
actual x/y/w batch against the earlier gradient diagnostic's batch receipt,
checks all saved logits/loss repeats, and verifies the float32 losses against
the earlier historical reports and saved originals. This authenticates inputs
and saved observations; it does not regenerate features or model activations.

Separate Python scalar `math.log1p`, `math.exp` and `math.fsum` code recomputes
dual weighted BCE, raw Huber consistency and nominal clipped-sigmoid gaps for
all 1024 samples. Group counts, sums, fractions and maxima match the published
independent vector reducer. Float64-to-float32 loss differences are reported;
the arithmetic comparison allowance is not a model/admission tolerance.

Actual result: Willow has 511/512 same-side nominal-saturated samples, contributing
99.97758522070366% of raw consistency loss; Nyx 509/512, contributing
99.99368675528351%. The remaining groups contain one and three samples. No
same-side saturated sample has a wrong binary label. This confirms that large
raw-logit consistency loss can mostly penalize nominally equal probabilities
in these original probes. It does not quantify final-model saturation, actual
complete quantized-head equality, full-dataset gradients, downstream recognition,
or the sole cause of the 35 original numerical failures.

```sh
python research/wakeword_p4/experiments/reduce_actual_raw_saturation_mac_v1.py \
  --archive "$PRIVATE/gpu-actual-raw-consistency-saturation-v1.tar.gz" \
  --output-dir "$PRIVATE/gpu-actual-raw-consistency-saturation-v1-authenticated" \
  --report "$PRIVATE/mac-actual-raw-saturation-independent-scalar-reduction-v1.json"
```

Run from the repository root in the authenticated native NumPy environment.
Output paths must be new. The committed report retains all eight cases,
input/observation identities, recomputed component losses and scope limitations.
No TensorFlow import, inference, optimizer, fresh development/acceptance, PCM IO
or board operation is involved.
