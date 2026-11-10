# Hey Nyx multi-TTS training results

Completed 2026-11-10 on an Apple Silicon M5 Mac. **Not accepted for production.**
No candidate from this experiment was flashed. The production panel continues
using Hi ESP, with its existing audio format and single microphone/AEC pipeline.

## Corpus and selection

Four engines contributed **9,317 source clips**: Piper 7,920, Kokoro 644,
Qwen3-TTS 480 and OpenAI TTS 273. This covers 945 fixed source speaker/preset
IDs, plus 32 Qwen description conditions that are not established independent
speakers. All Qwen clips are training-only. The 945 fixed source IDs do not
cross partitions; cross-engine physical speaker identity remains unknown.

Four correlated renderings per source produced 37,268 feature sequences.
Generators, splits, augmentation, source pins and exclusions are documented in
[README.md](README.md) and [generator-audit-v2.json](../results/generator-audit-v2.json).
Generator text supplies the labels: there is no claim that all clips passed
human listening or ASR. In particular, ASR confuses rare Nicks/Nix with Next;
it must not automatically relabel those positives as negatives.

The two models were trained on the same four-engine features. Checkpoint and
candidate choices used **float validation only**, before final quantized
test/regression scoring. Both exports use their best checkpoint, not the final
training step. All checkpoints and threshold grids are preserved.

| Candidate | Parameters | Training steps / selected checkpoint | INT8 bytes | Selected float validation mean source recall | Maximum source confusing false rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact | 16,225 | 40,000 / 10,000 | 44,504 | 54.56% | 2.08% |
| Official layer arguments | 25,537 | 20,000 / 10,000 | 63,032 | 64.74% | 0.625% |

Both selected checkpoints observed zero ambient events in 1.2336 validation
hours. The second model was selected because it met the observed false-rate
selection constraints. This is a selection rule, **not a production acceptance
decision**; it does not establish sufficient recall or a reliable hourly rate.

“Official layer arguments” means the notebook's layer configuration with this
experiment's existing 180-frame input. It does not mean the full official
duration/context/training recipe was reproduced. Training budgets also differ;
this comparison does not isolate a single cause of performance differences.

## Frozen quantized streaming result

Only the selected export was scored on the final regression/test sets. Its
threshold was recalibrated on quantized validation and frozen at **0.985**,
with a five-score sliding average and the evaluator's event latch/refractory.

| Measurement | Result |
| --- | ---: |
| Quantized validation positive recall | 62.40% (1,273 / 2,040 variants) |
| Test/regression positive recall | **62.11% (1,267 / 2,040 variants)** |
| Test/regression mean recall across the three evaluated TTS sources | 52.03% |
| Piper positive recall | 63.02% (1,210 / 1,920) |
| Kokoro positive recall | 41.67% (20 / 48) |
| OpenAI TTS positive recall | 51.39% (37 / 72) |
| Confusing-negative false detections | **0.30% (7 / 2,336 variants)** |
| Continuous ambient false events | 0 in 1.3896 hours |
| One-sided 95% Poisson upper bound for that exposure | 2.16 events/hour |
| Mac system-voice regression positive recall | **0% (0 / 50 variants)** |
| Mac system-voice regression false detections | 0 / 200 variants |

The float-window and quantized-streaming paths have different measured scores.
This experiment does not isolate quantization as the cause; the final artifact
is judged by its actual quantized streaming results. The zero Mac recall and
low cross-source recall reject it regardless of its lower observed false rate.
Changing a threshold alone is not an established solution.

## Artifacts and provenance

- [Frozen candidate comparison](../results/candidate-comparison-v2.json)
- [Compact training history and source hashes](../results/training-v2-compact.json)
- [Official-layer training history and source hashes](../results/training-v2-official64.json)
- [Quantized threshold selection](../results/threshold-v2-four.json)
- [Final quantized metrics](../results/trained-v2-four.json)

Selected model SHA-256:
`089cc964636749b549840cdb4202719bf394ca215f59f5e0a2414f8acb5b1614`.
Compact model SHA-256:
`e05ab9a2a02c5095a0f54d41760520f401c648c37d62a7deace6a21e95664fa8`.
Weights, generated/private audio, per-clip diagnostic transcripts and detailed
score arrays remain outside git. The compact run used the training script at
`ac3986e`; the second run used the added architecture/thread options. Each
training report records the exact Python source hashes used.

Earlier results are retained separately: [v1](../results/trained-v1.json) and
the [two-engine v2 core](../results/trained-v2-core.json). The latter observed
79.07% synthetic regression recall, 1.07% confusing false detections and 16%
Mac recall. Different data, renders and selection rules mean these are not a
controlled before/after comparison with the four-engine run.

## Limits and next evidence needed

The existing Piper/Kokoro and Mac sets have informed development. These are
regressions, not pristine final holdouts. The new OpenAI voice split was
reserved before generation; it remains a small synthetic test, not evidence
of real-user performance. Four render variants are correlated, and preset IDs
do not prove independent physical speakers. Environmental files are
blob-disjoint, but original room/session identities are unavailable.

There are no real-user positive recordings and no Chinese/TV/music negative
evaluation. Zero events in this short exposure cannot certify the project's
<=0.5/hour target. Recognition needs broader verified pronunciation coverage,
real-user/room evaluation and a closer examination of the upstream temporal
context/training recipe before integration. Full-app P4 flash/RAM/CPU, AEC,
reconnect and immediate post-wake speech checks are still release gates.
