# Original snapshot replay: mechanics pass, cross-platform numerics fail

Historical diagnostic, retained with its original failed comparisons. The user's
2026-11-15 clarification supersedes the proposed numerical follow-up below:
CPU/GPU training-gradient or logit agreement is not a qualification requirement.
Continue with actual wake recognition, false activations, latency and board
stability as described in the [current priority](../README.md#current-priority-usable-wake-detection).

The original GPU probe terminated successfully and delivered its actual
same-instance weights, variables, inputs and gradients. Mac anonymously
downloaded the 89,870,865-byte archive and verified all 170 payload files plus
the embedded manifest against GPU `f966d99`. This precedes learned execution;
the [intake receipt](../results/mac-engine-gradient-original-handoff-intake-v1.json)
does not claim inference.

Mac preregistered `bdca179` before the original graph run, then `f409cd0` before
the eager control. Original sessions 97582 and 84601 both genuinely terminated
exit 1 after all 100 cases. Neither was restarted after an observation timeout.
All arrays and failures are retained privately; complete per-case numeric
reports are published as lossless gzip with array hashes.

## Actual identities and mechanics

Both original H5 files load into the unchanged architecture with all 31 model
arrays byte-exact to their same-instance snapshots. The two original engine
variables also match, giving all 33 state arrays and all 17 trainable arrays in
the original order. All 19 upstream Python pins and the original four runtime
versions match. There is no reconstructed initialization or optimizer.

Each mode executes all 50 main/auxiliary pairs per word, twice. CPU repeat bytes,
None patterns, reversal-0 zeros, reversal-1 negative encoder gradients, unchanged
domain-head gradients and unchanged initial/end state all pass exactly. An
independent float64 softmax calculation from the saved GPU logits/labels matches
every saved domain-bias gradient within `5.282386034144437e-8` (prelocked limit
1e-6). This rules out a trivially fabricated all-zero bias gradient in these
cases; it is not a whole-network analytic autodiff oracle.

## Cross-platform mismatch, retained unchanged

Both CPU modes fail the fixed pointwise rule
`abs(delta)<=1e-4+1e-4*abs(saved)` in all 100 cases. Worst differences are:

| Quantity | CPU versus saved GPU, max absolute |
| --- | ---: |
| Main logits | 0.0022201240062713623 |
| Engine logits | 0.003215312957763672 |
| Losses | 0.0014524459838867188 |
| Supervised encoder gradient | 0.39274515211582184 |

The eager control uses the original GPU probe's execution style. CPU graph and
eager logits/losses are exactly equal across every case; their largest gradient
difference is only `3.814697265625e-6`. Thus graph-versus-eager execution does not
explain the observed CPU/GPU forward gap. This does not identify a unique TF32,
BN or kernel cause and does not demonstrate a recognition effect. No tolerance,
weight, case, coefficient or training budget was changed to obtain a pass.

The original mechanism claims and this failed cross-platform comparison have
different scopes: actual reversal wiring is consistent on both runtimes, while
independent numerical equivalence has not passed. Neither result establishes
usable wake recognition or actual P4 behavior.

- [Graph summary and full-report hashes](../results/mac-engine-gradient-original-graph-replay-v1.json)
- [Graph complete per-case report](../results/mac-engine-gradient-original-graph-replay-v1.json.gz)
- [Eager summary and all CPU-mode deltas](../results/mac-engine-gradient-original-eager-replay-v1.json)
- [Eager complete per-case report](../results/mac-engine-gradient-original-eager-replay-v1.json.gz)
- [Original plan](engine_gradient_replay_mac_plan_v1.md)
- [One-factor eager control plan](engine_gradient_eager_control_mac_plan_v1.md)

The historical proposed numerical diagnosis was an explicitly fixed original-case
CPU/GPU precision control at unchanged variables, without updates or new inputs.
It would record actual backend precision settings before attributing differences
to TF32. That proposal is not evidence that numerical drift caused the earlier
recognition failures. No new formal training, audio, reserved development/
acceptance, P4 write or production-board operation was performed on this Mac.
