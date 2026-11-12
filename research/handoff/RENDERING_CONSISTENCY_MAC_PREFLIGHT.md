# Independent Mac checks for the rendering-consistency proposal

These checks support a controlled experiment, not promotion of a wake model.
The GPU's proposed recipe is
[`3c075f5`](https://github.com/ChronoAIProject/nyxid-esp-panel/blob/3c075f5/research/wakeword_p4/experiments/paired_consistency_controlled_recipe_v1.md),
with the40-view class census corrected at`c3f89a9`:10positive and30negative
supervised windows,32quiet/rendered consistency pairs. Main128BCE plus paired
mean40BCE is common to both arms; only the consistency coefficient differs.
The inference graph, raw classifier input, existing selection and event gates
stay fixed. No numerical coefficient or expensive run is selected by this note.

## Original-array cross-ID measurement

[`measure_cross_id_scatter_mac.py`](measure_cross_id_scatter_mac.py) authenticates
the completed original-v46 Mac report and all2016saved-array files. It uses all
common absolute endpoints across all672IDs at each speed:16ends291..336at.9,
15ends297..339at1.0,11ends309..339at1.1, all step3. All four phrases and five
arms remain included. Sufficient vector sums give the exact mean over every
distinct-ID pair; no score-based pair sampling is used. Mean squared Euclidean
distance and cosine distance are separate metrics. Matching-ID near-word
comparisons and all distinct-ID cross-word comparisons are both retained.
Brute-force toy pairs independently check the sum identities.

The [completed result](../wakeword_p4/results/v46-cross-id-scatter-mac-v1.json)
shows mean distinct-ID Hey Nyx cosine distance.057367quiet and.184800noise15;
Nyx/Mix matching-ID distance is.979746quiet and.891629noise15. Cross-ID
Nyx/Mix distance is.981211/.900984respectively. These are endpoint-then-speed
means under this **all-ID intersection** scope, different from the earlier
per-ID-window weighting. Conditioning IDs are synthetic training sources, not
independent physical speakers. The result supports first isolating rendering
consistency without assuming extra target clustering is necessary. It does
not establish a cause, a beneficial loss or held-source generalization.

## Independent derivative and runtime checks

[`probe_rendering_consistency_gradient_mac.py`](probe_rendering_consistency_gradient_mac.py)
implements both-view derivatives independently in NumPy for
`D=mean32(1-u_quiet·u_rendered)`, `u=e/max(norm(e),1e-8)`.
The [executed report](../wakeword_p4/results/rendering-consistency-gradient-mac-v1.json)
records maximum ordinary finite-difference error1.1390170577552094e-10,
ID/phrase permutation and above-floor scaling invariance, no cross-phrase
gradient coupling, zero/equal/tiny/one-zero-view behavior, and counterexamples.
Both complete zero vectors and complete constant-unit embeddings are stationary
for D. A zero classifier head with weighted-prior bias also permits a stationary
constant embedding under BCE plus D. These checks explicitly **do not** prove
anti-collapse or optimization convergence. A nonzero-head example produces
opposite-label embedding gradients with opposite directions.

[`compare_rendering_consistency_mac.py`](compare_rendering_consistency_mac.py)
loads only the pinned GPU helper bytes at`c3f89a9` and compares TensorFlow2.21
autodiff to the independent NumPy derivative. The
[completed twelve-case report](../wakeword_p4/results/rendering-consistency-gpu-helper-mac-v1.json)
passes float64/float32 ordinary3D/full300D, zero, constant-unit, tiny and
one-zero-quiet cases. Formula correctness and a loss-helper check do not prove
actual-network head disconnection, unchanged BN/weights, identical main/paired
sampling or usable gradient scale. Those original-initialization50-batch
checks and fixed-coefficient execution receipt remain GPU prerequisites.

## Received original initialization arrays

The GPU completed the original50-batch probe at`5c0eaad`, with no optimizer
updates, reporting exact unchanged model/BN variables and disconnected head
gradients for D. Its [original four payloads](assets-gpu-paired-init-gradient-probe-v1.json)
have been downloaded and all sizes/hashes verified. The original CLI had exited
without saving an instance H5 or full variable inventory; this missing artifact
is explicitly retained, without recreating it as a historical snapshot.

The independent [Mac saved-array/source-window audit](audit_paired_init_arrays_mac.py)
has [completed](../wakeword_p4/results/paired-init-array-audit-mac-v1.json): all50
actual main128/pair40 tensor hashes, labels/weights and750per-variable gradient
records match. An independently indexed RNG schedule matches every ordered
pair source and endpoint. All2000paired windows are reconstructed directly
from the authenticated original uint16raw arrays and exactly match both their
original window hashes and saved scaled float32inputs. Main historical tensors
were not saved during v47; their historical equality is not claimed here.

Saved supervised gradients equal main+pair to maximum absolute error
3.814697265625e-6 under the original1e-5arithmetic tolerance. D's saved head
entries are exactly zero. Mac recomputation gives median encoder norms
256.3368631130569(supervised) and1.5408605787327252(D), R166.3595439140121,
and the preregistered coefficient**1.0**. The scaled gradient ratio spans
.0018603...0189365;17/50batches have negative gradient cosine. This is an
initialization-only scale observation, not later training balance or benefit.
No original-network autodiff is claimed by this saved-array audit. A later
reconstructed initialization can support numerical reproduction only after
checking original encoder/BN source values and classifier bytes, and must be
identified as reconstructed rather than an original same-instance snapshot.

All measurements are silent, use already examined training data or synthetic
numbers, and leave new development/final data and both physical boards alone.
V46/v47 remain rejected, v30 is retained, and matched Hi ESP recognition/FAR,
latency/resources and long-call/reconnect parity remain unproved.
