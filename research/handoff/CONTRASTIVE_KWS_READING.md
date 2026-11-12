# Contrastive KWS sources reviewed while original v47 trains

Read on2026-11-12. This is literature/source assessment, not a new training
registration, implementation or model acceptance. Original v47 remains unchanged.
The full objective still requires matched Hi ESP recognition/FAR/latency and
development-board resource/long-call/reconnect evidence.

## Source records

| Primary source | Original PDF identity | Relevant pages inspected visually |
| --- | --- | --- |
| Zhang, Zhu, Qin, [Margin-Aware Contrastive Regularization for Robust Streaming Keyword Spotting under Strict False-Alarm Constraints](https://www.isca-archive.org/interspeech_2026/zhang26ha_interspeech.html), Interspeech2026, DOI10.21437/Interspeech.2026-3545 | [Publisher PDF](https://www.isca-archive.org/interspeech_2026/zhang26ha_interspeech.pdf),1071887bytes, SHA256`a6390d4fb57d85df972bb3cc2cd1f4b9b415f63efcc8b8de4f5d94f21ef7de14` | PDF2-4: equations1-2, Figure1, setup and Tables1-4 |
| Ng et al., [I2CR: Improving Noise Robustness on Keyword Spotting using Inter-Intra Contrastive Regularization](https://doi.org/10.23919/APSIPAASC55919.2022.9980261), APSIPA ASC2022 | [Author preprint2209.06360v1](https://arxiv.org/pdf/2209.06360),1110585bytes, SHA256`a57a02dedfed21ef36c319a59666ae8e9b873b29a2e06150dde3a19ecb4204a3` | PDF3-4: printed loss, Figure3, setup and TablesI-II |

Crossref DOI metadata and primary publisher/author PDFs agree on titles and
authors. The I2CR version inspected is the seven-page September2022author
preprint; it is not asserted byte-identical to IEEE's final version. Original
PDFs and page renders remain in the local research evidence directory. Only
source links, hashes and review notes are committed, without republishing PDFs.

## What the sources actually support

MACR is closely related to the *kind* of failure seen here: closed-set accuracy
can remain high while strict streaming false-alarm budgets cause poor recall
against near words. It regularizes normalized penultimate embeddings during
training, adds a target-only supervised-contrastive pull and a cosine-margin
repulsion against nonmatching examples, and weights training-mined imposters.
It explicitly avoids pulling all heterogeneous Unknown examples into one class
cluster. The paper reports no extra deployed branch, parameters or compute.
These claims describe its own graphs, not a verified export of our model.

Its primary experiment is **ten target words plus Unknown/Silence**,64log-Mel
features and a150K-parameter causal1D-CNN, not our single Hey Nyx keyword,
40channel MicroFrontend or22801parameter model. It trains120epochs with
AdamW and weight decay1e-4; its main recipe uses temperature.1,margin.4,
hard-negative weight2and auxiliary coefficient.5. The top5000training negative
windows come from an EMA model and refresh every5epochs. None of these settings
has been selected or introduced into v47.

The event protocol uses a1second analysis window/10ms hop,50ms smoothing,
500ms refractory period and±200ms target matching. Test data consists of
50independently synthesized2hour streams,100hours total, with2000class-balanced
target events per stream. Validation thresholds are locked before test. Table1
reports FRR16.32→7.64%at nominal.5FA/h and29.85→14.80%at.2FA/h; these are
**paper-reported multi-keyword synthetic-stream results**, not our results or
the single-sided Poisson-upper-bound acceptance used in our studies. Raspberry
Pi4B latency and cross-backbone results do not prove ESP32-P4 timing, far-field
quality, Hi ESP parity or our long-call/reconnect behavior. The paper itself
leaves custom wake words and far-field conditions for future work.

I2CR supplies older, separate evidence for the noise-invariance hypothesis.
It pairs differently augmented versions of an utterance and other-speaker
examples of the same command during joint supervised/contrastive training.
The authors explicitly discuss same-source pairs retaining speaker/style bias,
which is relevant to avoiding a same-source-only claim of voice generalization.
It warms up without the regularizer in the first epoch and ramps its coefficient
up to.5. In its35command test set, EfficientNet-B0 traffic/-10dB accuracy moves
.840→.858with intra-only and→.867with inter+intra; this modest, specific example
must not be described as a wake-FAR improvement.

That paper evaluates10or35closed-set Google Speech Commands,64FBank features,
EfficientNet-B0/ResNet18/KWT1and classification accuracy under FSD50K/noise
augmentation. It does not establish event-level near-word false alarms at.5/h
or independent custom-wake/MCU behavior. Its noise, circular shifts, cropping
and speed augmentation are paper settings, not changes to our PCM or audio
pipeline. In particular, its printed contrastive denominator is indexed by
`n not in P(i)`; it does not explicitly resolve all self/anchor conventions.
Do not silently substitute a standard all-nonanchor SupCon denominator and
call it an exact reproduction without a separate implementation contract.

## Single-keyword incompatibility checked before proposing a copy

MACR equation1 restricts its denominator to other **target-keyword** examples.
With only one target label, positive matches equal the entire denominator.
For two target examples, each denominator has just its one positive; the log
ratio is identically zero at every angle. Unknown examples in the same batch
are excluded. Thus this pull component alone supplies no attraction for that
case. With three examples, collapsed embeddings and a120degree regular simplex
have the same uniform-positive loss3log2, despite very different compactness.

The [executed analytic/numeric probe](../wakeword_p4/results/single-target-pull-probe-v1.json)
reproduces both identities from the printed equation, with angular finite
differences in the two-example case. It does **not** test the complete MACR
objective: separate CE and margin-push gradients can still change embeddings.
The published multi-keyword experiment does not have this single-target
specialization, so this is a compatibility limit, not a failed reproduction
or a claim that the original paper is wrong.

```sh
python3 -B research/handoff/probe_single_target_pull.py \
  --output /tmp/single-target-pull-fresh.json
```

## Consequence for the next decision

The literature supports considering a **training-only representation objective**
after original v47 has a complete result. It does not justify copying a loss
unchanged, adding another concurrent training run, changing the deployed graph,
or overriding rejected-model/unused-development boundaries. It also is not the
same objective as the already failed same-voice scalar-score ranking, four-way
auxiliary CE or phone KL/CTC recipes; those failures remain recorded.

If v47 fails, first inspect the frozen actualfloat same-source clean/noisy
representation displacement and matched same-ID near-word separation across
the complete training grid, preserving all sources. A future objective would
need explicit single-keyword positive/negative masks, temporal alignment,
normalization/reduction and empty-set behavior. It must preserve Unknown
diversity instead of clustering all negative speech, and separate improved
source fit from held-source/engine generalization. Gradients and a matched
binary-only control would need concrete checks before any formal study.
No formula, coefficient, new hard-negative mining, weights or training process
is chosen by this reading note. Existing event gates and matched Hi ESP/P4
requirements remain the basis for success.
