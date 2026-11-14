# Published one-way formal source review on Mac

The actual read-only review at GPU commit `1fe7c22` passed 558 unique public
file bindings, the complete smoke/formal update-step syntax comparison and all
seven validation namespace derivations. The original initialization, optimizer,
batch-schedule binding, export and runtime-version metadata remain identical.
The one-way helper passes only `float_bounded` through `stop_gradient`; both raw
logits still enter the original supervised loss. The original numerical gates
are unchanged. No running GPU source was edited or restarted.

```sh
python3 research/wakeword_p4/experiments/review_oneway_formal_sources_mac_v1.py \
  --repository-root . \
  --gpu-ref 1fe7c22 \
  --report NEW_REPORT.json
```

The published Python 3.11 complete-step AST digest is reproduced with Python
3.12.13 after omitting only its added, empty `type_params` fields. Source hashes,
full AST equality and the declared digest are checked separately. Formal has one
10000-update instance per arm; the two fresh repetitions cover only the four
smoke updates. This review makes no additional full-training repetition claim.

The continuation plan binds PID 1450872, start time 22669355 and the original
command line. Those are authenticated plan fields, not a Mac observation that
the GPU process is currently live. The smoke result/audit hash chain is checked
as published metadata; private H5, Adam and batch arrays are not independently
read here. This source review proves neither training completion nor learned
numerical, recognition, FAR or independent-acceptance quality.

The old event-ledger request remains separate. The frozen `9162bdb` reader still
requires the original source manifest and all 1242 event NPZs. The metadata-only
Willow finding at `38c36fa` does not replace that intake or qualify any model.
