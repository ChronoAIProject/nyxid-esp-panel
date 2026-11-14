# Original official/custom PCM contracts and output semantics

The actual JSON-only comparison completed with exit0. It joins immutable GPU
`f312c299d4f33c6347e52eb17985a0815c0a8d7f` execution plans, old collection
metadata, the original synthetic manifest already authenticated on Mac, and
the official background PCM contract. It does not open WAVs, model weights,
scores or new development materials.

- All360 original synthetic rows are exactly equal, including order, encoded
  and decoded hashes, sample counts, source/target roles and generated bounds.
- All372 encoded WAV identities match both execution locks. All12 background
  decoded PCM identities/counts match their original input contract. The same
  synthetic/background manifest hashes are bound by original official, old
  fixed-grid and earlier custom execution plans.
- All1,860 collection records cover the five old models ×372 complete inputs.
  Original sample/frame/padding/step accounting and cross-model raw-feature
  hash identities agree. This checks saved metadata, not a repeat frontend run.
- Background exposure is603,875,222 samples at16kHz:10.483944826388887 hours.
  The original official grid is2 models ×13 requests ×372 inputs =9,672 streams.

The original360 set has36 Willow positives,36 Nyx positives,36 Hi ESP positives
and252 common negatives. It is already consumed diagnostic evidence. Future
396-row Jenny/Muse development is separate and is not authorized or read here.
The eight source bindings in GPU's same-recipe inventory are also verified.

This establishes matching **declared input contracts** and complete exposure;
it does not independently decode or prove every original detector execution.
The official calibration/native-return results are not part of this read.

## Keep training-condition names separate from output heads

Reading the execution-bound collector reveals a necessary qualification to
the earlier shorthand “Willow float-control float any-event7–8/36”:

| Path | Actual observable |
| --- | --- |
| Old four H5 models, `kind=float` | `bounded(model, tf)` float stream, nominally sigmoid(clip(raw Dense logit,−8,8)) |
| Old four H5 models, `kind=uint8` | Original fixed-grid compiled integer artifact and its bounded head/posthead |
| Proposed new FP32 paired route | Original Dense sigmoid and unchanged BN, without the bounded head |
| Official reference | WakeNet9 native detector returns and actual threshold getter, without custom five-score smoothing |

The source H5 training-condition label `float_control` does not mean the old
collector measured raw-sigmoid inference. The collector explicitly invokes
`to_streaming_inference(bounded(model, tf), ...)`. The helper replaces the head
with ReLU6 Dense `W*3/8, b*3/8+3`, then Rescaling`8/3,−8` and sigmoid.
Both source identities are bound by the original old-regression plan:

- `collect_old_fixed_grid_regression_v2.py` SHA256
  `e7f7b230f885957550175c168181d2788937dabdddf80f4d4c0c12f5442b4ba7`
- `probe_hey_broad_bounded_float_v1.py` SHA256
  `b4171e46a30d87f23897bc40b26e1839cee7efa8f090359bb1e2a1dc5b669f9b`

Thus the old7–8/36 count remains valid for the old bounded-float observable.
It is **not** a measured count for the prospective raw FP32 stream. In exact
arithmetic the head clipping changes a probability by at most sigmoid(−8),
about.00033535; affine evaluation roundoff and event-state/refractory effects
still prevent transferring event counts without the proper stream experiment.
This theoretical clipping bound is not a measured whole-stream equivalence
certificate. It neither rescues the old rejected integer artifacts nor proves
the original raw model is good or bad enough for deployment.

Keep the FP32 scope as raw-output numerical conversion diagnostics, followed
by its own admitted recognition/FAR evaluation. When the original official
curves arrive, label each original observable accurately and compare only
common-FAR diagnostic points with these same inputs. Preserve null/failed
points and actual-getter DEFAULT/SET duplicates. No best-point promotion or
independent acceptance conclusion follows from this historical comparison.

## Reproduce the contract join

```sh
PYTHON research/wakeword_p4/experiments/compare_old_official_pcm_contracts_mac_v1.py \
  --repository-root REPO \
  --synthetic-manifest ORIGINAL_HEY_DEVELOPMENT_REPLAY.json \
  --report NEW_REPORT.json
```

The result is `results/mac-old-official-pcm-contract-comparison-v1.json`.
The code uses only stdlib JSON/hash/git reads and refuses report overwrite.
No playback, microphone, USB or production-board operation is involved.
