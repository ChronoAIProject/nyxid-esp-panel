# Engine-input construction: executable review and real receipt intake

The engine-adversarial hypothesis is distinct from the failed frozen-encoder
experiment, but source-engine causality is not established. Before interpreting
its gradient probe, this review executes the actual unchanged input producer and
auditor from GPU `c724230` in an isolated constructed fixture.

The fixture has 297 entirely constructed feature rows: eleven texts, three
engines, three variants, and three source geometries. Its private git repository
has no remotes and authenticates only a labelled synthetic lock. No admitted
training features, audio, model, optimizer or GPU process is accessed. The Mac
NumPy version is 2.4.3, distinct from the GPU's pinned 2.4.6.

The original producer and auditor both actually exit 0 on 50 batches / 1,650
constructed windows. A separate row-local coordinate reconstruction agrees
exactly with every saved phase tensor and checks all 33 lexical/engine pairs per
batch, variant cycling, clipping and the zero final frame. It exercises 562
singleton-end windows and 531 lower-bound-225 windows. Reexecution produces the
same report bytes.

## Confirmed auditor coverage gap, bounded impact

Two deliberate chronology corruptions are accepted by the original auditor:

| Constructed mutation | Original auditor |
| --- | --- |
| Batch 4 step changed from 4 to 1 | exit 0 |
| Batch 1 step changed from 1 to 52 | exit 0 |
| Incorrect entry source identity | exit 1, correctly rejected |

The first two preserve `(step-1) % 3`, so variant and RNG/tensor checks still
agree. The auditor needs an explicit ordered `1..50` step check if it claims to
authenticate batch chronology. This is a validator coverage gap, not evidence
that the actual GPU tensors or order are wrong. A narrow additive audit of the
existing result is sufficient; it does not require regenerating inputs or
restarting a gradient job pinned to the old auditor.

## Actual published result metadata

The separate intake checks real GPU `518c72e` input result SHA
`99c3f1a5c3aff692c68de6a833598bfc1e5ba139cd83af4df3895ecbf6902fef`,
execution and independent-audit bindings, and all seven public source/report
pins. All actual batch steps are correctly ordered `1..50`. Every batch has
all 33 text/engine pairs exactly once with its fixed common variant, declared
legal endpoints and the expected array filename. All 99 cell populations are
nonempty. `Hey Nicks.` is the documented positive pronunciation for Hey Nyx;
both positive words are present.

The original raw metadata and binary feature files are private. This intake does
not independently replay their RNG/tensors or verify generated boundaries against
audio. Their audit remains attributed to the GPU's actual execution. No input
feasibility result proves engine invariance, gradient correctness or recognition.

The first local run reached report serialization after execution and controls,
then failed because a NumPy integer accumulator was not JSON serializable. That
attempt is retained privately; only the accumulator conversion was corrected
before executing in new directories.

- [Executable review](review_engine_input_builder_mac_v1.py)
- [Complete report, source hashes and controls](../results/mac-engine-input-builder-constructed-review-v1.json)

```sh
python3 research/wakeword_p4/experiments/review_engine_input_builder_mac_v1.py \
  --gpu-ref c724230 --actual-ref 518c72e \
  --output-dir /private/path/new-engine-input-review
```

The goal remains active. Original data-quality, independent-development,
recognition, numerical and P4/audio-lifecycle requirements are unchanged.
