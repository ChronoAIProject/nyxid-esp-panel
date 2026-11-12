# Next hypothesis: same-source phrase ranking

Status: proposal, not a started run. Both completed consistency arms remain rejected. No new development or final data may be read to develop or rescue this proposal.

## Evidence and hypothesis

The completed matched-input experiment produced no eligible point in either original138-point grid. At the same threshold .99609375, treatment negative false clips decreased6→1 while macro recall decreased .39166666666666666→.25. This supports testing discrimination directly; it does not prove global score suppression is the sole cause. Full-grid same-threshold audit retains every point without selecting an alternative threshold.

Use the existing authenticated training-only four-phrase/five-rendering pack. For every conditioning ID, speed, rendering and common endpoint, put the target Nyx scalar logit and Nick/Next/Mix scalar logits in one group. Add listwise phrase loss `logsumexp(z_Nyx,z_Nick,z_Next,z_Mix)-z_Nyx`, averaged over10 ID/rendering groups. This explicitly rewards the target relative to near words within one source. No cross-ID attraction, new embeddings, deployment head, margin, temperature, score mining or filtering. The unchanged scalar sigmoid head remains deployment output. This is a different intervention from rendering consistency and has no D term.

Proposed matched design: original paired control recipe with coefficient0 versus identical recipe plus phrase ranking coefficient1. Coefficient1 is fixed by definition before gradient observation, not chosen from old/new development metrics. Keep both main128 and paired40 BCE, label weights, original initialization, deterministic numerical policy, frozen BN, input draws, one Adam update per step,120k updates and original checkpoint/PTQ/old138 gate. Explicitly observe ranking backward in both arms; head gradients are expected and must be checked. Save all actual preupdate initialization and complete traces. No sweep/restart. Reusing the completed control as a new matched control requires proving unchanged execution inputs and per-step observations; default is not to claim it automatically qualifies.

## Mandatory bounded preflight before formal commands

1. Independently verify stable logsumexp value and analytic gradient softmax(z)-onehot(target), including equal/large logits, shift invariance and finite differences. Equal logits must give log4 and nonzero label-directed logit gradients. This is not proof against stationary collapse of the complete network.
2. Verify all10 groups contain exactly one target and three distinct intended negative phrases from identical source/speed/rendering/endpoint. Authenticate raw source pack, no validation path, exact independent schedule and label mapping; never infer phrase order from batch position without checking metadata.
3. Replay saved actual first50 training batches at authenticated initialization without optimizer updates. Save per-group logits, BCE/ranking loss, full encoder/head gradients, norms and cosine. Check finite values, connected gradients, no variable/BN mutation. Observe scale only; do not change coefficient from this probe. If unstable or disconnected, stop and report instead of silently retuning.
4. Fresh matched2-update smoke with exact first-step inputs, initialization and observations; finite weights/frozenBN/export contract, one optimizer update per step. Preserve failures. Deliver concrete execution pins and fixed commands through Issue13 before expensive runs. This is engineering sequencing, not a permission request or requirement to wait for optional feedback.

If either arm passes the unchanged old-stage gate, it still needs the untouched-development and reserved-final synthetic protocol and matched official Hi ESP comparison. If both fail, reject both and retain v30. This proposal is not an assertion that ranking will meet these gates; do not start formal training until preflight proves the implemented experiment matches this specification.
