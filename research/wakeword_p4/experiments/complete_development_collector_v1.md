# Complete development raw collector v1

`collect_complete_development_v1.py` collects whole-input raw custom-model traces using `complete_feature_stream_v1`. It leaves historical scorers/results unchanged and performs no calibration, threshold selection, official-model scoring, acceptance or promotion. It has no hardcoded candidate directory, model byte size or material count.

## Execution record

The GPU producer creates and pushes a new execution JSON from actual completed artifacts. Its schema is:

```json
{
  "schema": "complete-development-stream-execution-v1",
  "tail_policy": "whole-features-final-zero-pad-stride3-v1",
  "authorizes": "development-raw-collection-only",
  "versions": {
    "numpy": "ACTUAL", "scipy": "ACTUAL", "soundfile": "ACTUAL",
    "ai-edge-litert": "ACTUAL", "pymicro-features": "ACTUAL"
  },
  "files": {"EVERY_BOUND_PATH": "SHA256"},
  "materials": {
    "synthetic": {"path": "MANIFEST", "sha256": "SHA256", "role": "reused-development"},
    "background": {"path": "MANIFEST", "sha256": "SHA256", "role": "reused-development"}
  },
  "models": {
    "MODEL_ID": {"path": "MODEL", "sha256": "SHA256", "bytes": 1, "output": "NEW_DIRECTORY"}
  }
}
```

`bytes: 1` above is a placeholder, never a fixed model-size rule. Each model entry must contain the actual frozen export size/hash. The record producer admits fixed-final training/model evidence according to the broad scoring contract and binds it in `files`; the collector verifies bound bytes but does not replace training admission or infer provenance from a model's size.

Both material roles must explicitly be `new-development` or `reused-development`. This raw collector records either role; the active experiment contract determines which may be used for threshold selection. GPU comment `6866411177` reports its direct user objective explicitly requires unconsumed development for threshold selection, so reused inputs there remain regression evidence. Neither role label proves source independence; final success still needs independent acceptance. Each material is a JSON object with a nonempty `records` list. Synthetic records retain the existing `path`, `sha256`, `samples`, `pcm_sha256` fields. Background records retain `wav_path`, `wav_sha256`, `samples`, `pcm_sha256`, optionally `frontend_path`/`frontend_sha256`. Other frozen labels, boundaries, conditions and source metadata pass through unchanged.

Bind both manifests, every WAV/cached frontend, every model, this collector, the complete-tail helper, original `uint8_stream_detector_v1.py`, `training/features.py`, and all installed files returned by `runtime_identity_files()`. That list explicitly includes the native `micro_features_cpp` extension outside the Python wrapper package. Preserve additional material-history/model/event/metric/training evidence required by the scoring contract in the same record. Canonical-equivalent filesystem paths are accepted only when their exact file digest agrees.

The collector requires local lock bytes to equal the specified published Git ref. Versions, file digests, material counts/sample declarations, explicit roles and model sizes/hashes are checked before runtime construction. Run once per registered model:

```sh
python3 research/wakeword_p4/experiments/collect_complete_development_v1.py \
  --execution-lock research/wakeword_p4/results/NEW_EXECUTION.json \
  --model MODEL_ID
```

## Output and invariants

Each original WAV is rehashed and independently checked after decode as complete 16kHz mono PCM16. The whole frontend runs once or its authenticated whole cached output is used. The detector resets only at complete-source start; a second fresh-reset traversal must match every raw output and clock. Final feature padding is explicit, and no real feature is dropped. The inherited detector still verifies the actual INT8 `[1,3,40]` / UINT8 `[1,1]` interface and output scale.

Per-source NPZ contains raw UINT8 outputs and original sample-support clocks. `records.jsonl` contains the complete original row, trace digest, tail metadata and reset-repeat result. The final report binds the execution/model/materials, each phase log and all trace digests. A model's registered output directory must not exist; partial or completed results are preserved on failure, with no automatic retry or cleanup. v30 under this new tail policy needs a new output directory and retains its old results separately.

## Constructed regression

`review_complete_development_collector_v1.py` uses three temporary numerical PCM16 WAVs, independent standard-library WAV decoding, synthetic features and a toy stateful detector. It tests a nonhistorical 1,004-byte model, published-lock/file bindings, canonical aliases with wrong-hash rejection, complete tails/cached background and reused roles, output preservation, same-size model corruption, changed publication bytes, acceptance-labelled phase rejection, reset-dependent output rejection and decoded-PCM mismatch. Publication/package/native-runtime boundaries are mocked; byte checks and collection run normally. These are workflow checks, not learned-model/frontend/runtime/P4 or recognition results.

The initial constructed run failed before runtime construction because macOS `/var` and `/private/var` named the same temporary helper with different absolute spellings. That failure is retained in `results/mac-complete-development-collector-initial-failure-v1.json`. Canonical-equivalent lookup fixes the path comparison without weakening digest verification; the wrong-hash alias regression passes. The complete successful result is in `results/mac-complete-development-collector-review-v1.json`.

```sh
python3 research/wakeword_p4/experiments/review_complete_development_collector_v1.py \
  --gpu-ref 9fd2d76 \
  --output NEW_PATH.json
```
