# Complete existing cross-engine training support census

After the frozen MatchboxNet experiment fit every training bare target (63/63) yet missed every consumed Melo bare target (0/18), the next question is whether there is already admitted cross-engine lexical support for this distinct pretrained representation. This is not a claim that increasing source count will work: earlier small-encoder source-coverage and lexical interventions failed and remain binding evidence.

The complete existing training metadata contains **37,944** bare rendered records for Hey Nicks / Hi E S P / Hey Nick / Hey Next / Hey Mix / Hey Muse, across Piper, Kokoro and Qwen. Every selected file exists and its full file SHA256 matches original metadata; all original renderer quality flags are true. No waveform was synthesized, rewritten, trimmed or selected by duration. Source fields, exact container bounds and feature-to-renderer identity were checked for every selected record. Conditioning identity counts are not person counts. This audit does not rerun the original native-float quality tests or establish physical-speaker/pretraining independence.

| Engine | Each ordinary class | Nyx | Conditioning IDs per class |
|---|---:|---:|---:|
| Piper | 6048 | 6048 | 672 |
| Kokoro | 180 | 180 | 20 |
| Qwen | 72 | 216 | 24 |

There are **no missing files, hash mismatches or failed original renderer quality flags**. The complete original training metadata owner, renderer metadata owner and audit script hashes bind this census. Existing original role exclusions still require explicit binding before any new run; this report alone is not a new source admission or training execution lock.

A concrete task-support issue prevents naive reuse of the failed 20,480-sample recipe. Kokoro complete spans often exceed that context: 158/180 Nyx renders and all 180 Hi E S P renders cannot fit with the prescribed maximum 1024-sample after-source offset. Qwen also contains overlong inputs. The largest complete source+RIR span is **43,760 samples**. The smallest 160-aligned window preserving every source plus that offset is **44,800 samples (2.8 s)**. These are full synthetic containers, not phoneme measurements. Filtering to short utterances or cropping long utterances would change the proposed coverage test and is not authorized as an implicit repair.

The next bounded experiment, if its source and resource audit succeeds, should compare a **same 44,800-sample window SpeechT5-only control** with a **SpeechT5+Piper+Kokoro+Qwen engine-balanced treatment** from the same original pretrained initializer, using exactly the same fixed optimizer, updates and final-only export policy. This can isolate the pooled source/task-support package at a common window; comparison to the old 20,480-sample run alone cannot. Preserve all sources, use explicitly balanced engine/class draws, and disclose unequal native Qwen positive draw counts. Do not quietly add commands without trustworthy source bounds.

Before execution, bind original native quality/provenance and conditioning-role audits, inspect the increased frontend/neural cost and memory, then freeze both recipes and consumed-only diagnostic policy. Do not reopen Melo India/Australia, Mike, original final sets or reserved backgrounds to rescue this proposal. No threshold, seed, checkpoint or window sweep. A promising consumed diagnostic would still need genuinely unconsumed development and independent qualification against official Hi ESP; this is not a promotion plan based on reused scores.

Reproduce the audit with `research/wakeword_p4/experiments/audit_matchbox_multiengine_support_v1.py`; result: `research/wakeword_p4/results/4090-matchbox-multiengine-support-audit-v1.json`. No new model trained, no recognition calls and no development/acceptance audio accessed. Frozen v30 remains unchanged.
