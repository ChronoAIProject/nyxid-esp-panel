# FP32 internal-state software stage

Mac currently has no development USB. P4 resource/whole-panel evidence remains
required for deployment, but is not a dependency of export and host numerical
diagnostics. The earlier ordering that waited for P4 before any trained-model
FP32 export is revised for this software-only stage. No fresh development or
acceptance scoring, new training, board operation or candidate promotion follows.

## Source compatibility finding

Actual pinned `microwakeword/utils.py` SHA256
`7d0c9200079772302a5320944c7d3b93ec4e7a83ee560e1734a3e8d8d2bb8717`
supports only internal-state and non-stream inference in
`convert_to_inference_model`. The remaining branch raises
`ValueError('non supported mode ', mode)`. An external-mode enum and layer
methods exist, but do not implement the missing factory conversion/adapter.
GPU `d652049b` is an unexecuted prospective external-state scope, not an
observed execution failure. Its exporter has not yet been implemented/locked.

The actually validated Mac MicroInterpreter probe uses one float32[1,3,40]
audio input and one probability output, with internal resource variables. It
does not feed/carry six external state inputs. Its source SHA256
`249a548d484ef6f59ddbc87f340edbd0e3497a33292af89e68f5d71c255c21e7`
matches the original frozen host plan. That evidence cannot certify a new
external-state ABI. See `mac-fp32-state-mode-scope-source-audit-v1.json` for the
actual source/AST checks; no model or media was accessed by that audit.

Reproduce the source inspection without importing an ML runtime:

```sh
python3 research/wakeword_p4/experiments/audit_fp32_state_mode_scope_mac_v1.py \
  --repository-root REPO --upstream PINNED_MICROWAKEWORD --report NEW_REPORT.json
```

## Next software execution scope

Prefer the same original internal-state factory already validated by the random
FP32 export, complete P4 link and real host TFLM run. Include all four original
paired final10000 H5 identities from `d652049b`, not selected checkpoints.
Preserve original Dense sigmoid/BN/parameters and all six fresh-zero internal
states. Export FP32 builtin-only without quantization or bounded-head changes.
Publish actual exporter/runtime/input/model identities before executing.

First use the **existing** four synthetic feature streams, not old135 requests
or recognition/development/acceptance audio: zeros, constant26, ramp257 and
xorshift32, 256 steps each, in that order. Original `original-host/inputs.bin`
is491,520 bytes, SHA256
`0f7eff41940be4149be682f9a24755d42646bbe08e211c5fbdd1638059d3c9ac`.
It is included in the already published436,399-byte host artifact:

https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/mac-fp32-real-tflm-host-v1.tar.gz

Archive SHA256:
`45f25e7d495c732fd764344d181cc47717fa712fcdbac0af261e3b50e9c442dd`.
The original random-model outputs in that packet are **not** references for
trained weights. GPU must create separate raw TF and LiteRT references for
each of the four exact H5s, two fresh/reset repeats per stream, all startup
steps retained, finite outputs, repeat bytes exact and max probability
difference<=1e-4. Save originals, including any failure; no outcome-based retry.

Deliver four actual TFLites, metadata and complete TF/LiteRT references, binding
the original H5 hashes, full variable inventories and six initializer states.
The consumer must independently require exactly one audio input/one probability
output, FP32 tensors, permitted operators and six real-zero initializers before
using the internal-state host runner. That runner alone does not validate a
multi-input ABI. Mac then freezes the trained-model host run and checks actual
MicroInterpreter outputs on those same bytes; no original random run is redone.

This is an intermediate software check, not the full540 numerical qualification
or recognition test. The external-state proposal is retained as unexecuted;
its state-output tolerance is not asserted as already implemented or verified
by this single-input stage. Full-stream phase/numerical validation requires a
separate actual execution lock for the chosen representation. Existing integer
rejections, fresh-development admission, shared-FAR evaluation, independent
acceptance, P4 resource/audio/first-utterance/long-call requirements all remain.
