# Positive target-support intervention (proposal)

Status: proposal only. No formal training or model promotion is authorized by this file.

The completed phrase-ranking experiment improved same-threshold discrimination but both arms failed the unchanged 138-point joint gate. At the fixed historical threshold `0.99609375`, treatment changed the saved synthetic result from 25/76 target streams and 6 false clips to 35/76 target streams and 2 false clips. This leaves a large absolute positive-coverage gap.

The authenticated training-only support audit gives a concrete, separate hypothesis. Complete original PCM support exists in only 509,618 of 1,500,000 Piper candidate draws (0.3397453333), 56,397 of 75,000 Kokoro draws (0.75196), and 5,130 of 7,500 Qwen-CUDA draws (0.684). Long positive prompts are especially uneven: `Hey Nyx, check the volume.` has complete source support in 331/815 Piper sources, while the short `Hey Nyx.` has 2,585/2,585. These counts describe sampler coverage only. They do not prove a pronunciation, target-phone interval, receptive-field, or label error; full-source support is not a substitute for verified keyword timing.

## Preflight

Use only the existing authenticated training pack. Do not read validation, new-development, or final data to construct or tune the intervention.

1. Define and authenticate a target-interval support predicate from source metadata, verified target-phone boundaries, feature offset, context length, endpoint, and source hash. Keep full-source support as a separate diagnostic column. Do not infer phrase timing from array position or treat a long command's full-source failure as a label failure.
2. Enumerate all legal endpoints for every source group before any gradient is computed. Record `target_supported`, `full_source_supported`, and `no_legal_endpoint` separately, including the exact reason and counts. Preserve the original source, phrase, speed, and rendering populations in both arms.
3. Predefine the no-legal-endpoint behavior: retain the original sampler draw unchanged and mark it `unsupported_fallback`; never silently drop, reweight, pad, truncate, or change speed. The treatment may choose a deterministic legal endpoint only for a source group with at least one authenticated target-supported endpoint.
4. Run the first 50 training batches at the frozen initialization with no optimizer updates. Save the original and support-aware positive-window ledgers, fallback counts, BCE losses, connected gradients, and all mutation/BN checks. This is a scale observation, not a coefficient or threshold search.
5. Run a fresh matched two-update smoke. Control uses the current sampler. Treatment applies only the predeclared legal-endpoint substitution above. Negatives, architecture, scalar deployment head, frozen BN, audio speed, PCM, and normal single-stream inference remain unchanged. Audit exact initialization, ordered draws, and model export.

## Formal experiment if preflight passes

Run one fresh control and one treatment, each for 120,000 updates, with the original checkpoint interval, post-training selection, PTQ, and fixed 138-point INT8 gate. Keep the original main128 and paired40 BCE terms and all weights. The only treatment difference is the positive support-aware sampler. Both arms must log full input draws and loss rows, and both must use one ordinary Adam update per step. No automatic restart, threshold rescue, score mining, coefficient sweep, or new development/final access.

Reject both arms if no eligible point exists. A passing historical gate still requires the untouched development/final protocol, Mac replay, P4 numeric/resource checks, matched Hi ESP acoustic checks, and long-call/reconnect stability before any firmware change.

The proposal does not claim that source support is the cause. It defines a falsifiable intervention while preserving the existing audio path and acceptance boundary. If the target-interval metadata cannot be authenticated or most sources have no legal endpoint, stop at preflight and report that result instead of launching formal training.
