# Training witnesses and persistent confusion timing

The already-audited Nyx float diagnostic has 13 common-negative clips that
trigger at every original registered threshold. Independent reduction of all
1,794 first events shows that every first trigger is after the original
generated prefix boundary, by 4.625–578.5625 ms. Therefore these observations
cannot be reduced to pre-boundary triggering alone. Generated boundaries remain
synthetic allocations, not measured phonetic ends. The source/report are
`audit_fp32_confusion_timing_and_sampler_mac_v1.py` and
`../results/mac-fp32-confusion-timing-and-sampler-source-v1.json`.

The original paired-QAT execution lock authenticates the inspected sampler and
runner. Positive crop ends are keyword-frame +3 through +27; explicit own-prefix
negatives end -35 through -15. The -14 through +2 gap is a policy observation,
not a demonstrated bug or unique explanation of the post-boundary confusions.
Each batch retains 32 positive, 32 ordinary negative, 16 prefix negative and 48
ambient slots. Negative loss weight is 20, positive weight 1; the aggregate
negative/positive weight-mass ratio is 60, not an empirical class prior. The
paired runner uses no mining or spectrogram masking. Static policy and surviving
positive checks do not measure actual effective per-phrase input frequencies.

## Saved training-witness event diagnostic

`score_original_training_witness_events_mac_v1.py` evaluates the original saved
TF and LiteRT probability trajectories, not models. Input contracts are the
previously authenticated full 135-request training packet and the original
138-point raw-FP32 grid. All startup, cold-start state and complete original
clocks remain; no cropping, re-export, new inference, source selection or
workpoint selection occurs. Both backends are scored and event differences are
retained rather than assumed absent from a small probability error.

The first scoring attempt at `efb731f` stopped before reading probabilities or
creating an output directory: its reader incorrectly required original request
orders to be consecutive. The 135 distinct retained orders extend to 106334.
The additive `mac-original-training-witness-events-plan-v2.json` fixes only this
reader contract: array positions and original request orders are checked
separately, without renumbering or changing inputs. The original plan/source and
`mac-original-training-witness-events-initial-setup-failure-v1.json` remain.

The v2 reader later stopped on an emitted common-negative event: all 81 original
negative requests have a container end but no generated-prefix annotation. Its
828 complete point records are preserved in the original partial ledger, with
identity in `mac-original-training-witness-events-missing-boundary-reader-failure-v2.json`.
The additive v3 plan preserves missing prefix offsets as null and validates the
54 annotated positive / 81 unannotated negative contract before scoring. It
does not invent a prefix boundary, modify original clocks or remove events.

Each model has 27 own-target derived examples (Piper 12, Kokoro 12, Qwen 3),
27 reciprocal-word examples and 81 common negatives. These counts include
correlated variants. They are training witnesses, not independent voices or a
generalization set. The integer-trained raw reconstruction is also a different
observable from its original quantized training execution. No FAR is computed:
there is no independent background exposure in this subset.

First publish the `plan` output and source, then run that exact plan. The plan
binds the original intake/audit, packet request/grid bytes and code hashes.

```sh
python research/wakeword_p4/experiments/score_original_training_witness_events_mac_v1.py plan \
  --repository-root REPO --training-packet ORIGINAL_TRAINING_PACKET \
  --development-packet ORIGINAL_REUSED_DIAGNOSTIC_PACKET --plan NEW_PLAN

python research/wakeword_p4/experiments/score_original_training_witness_events_mac_v1.py run \
  --repository-root REPO --training-packet ORIGINAL_TRAINING_PACKET \
  --development-packet ORIGINAL_REUSED_DIAGNOSTIC_PACKET --plan PUBLISHED_PLAN \
  --output-dir NEW_PRIVATE_DIRECTORY --report NEW_PUBLIC_REPORT
```

All 1,104 model/backend/grid summaries and 149,040 case/point event records are
written to the private gzip JSONL ledger. The public report retains its hash and
all-region ranges by engine and common phrase; it selects no threshold. Actual
saved events can distinguish observed failures on these training examples from
the already-known development failures, but cannot certify the full training
distribution, a unique root cause, independent acceptance or P4 qualification.
