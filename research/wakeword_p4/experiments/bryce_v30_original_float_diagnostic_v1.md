# Consumed126 original-v30 float diagnostic

This diagnostic compares the authenticated original 70000-step SavedModel,
the same float graph with input quantized/dequantized using the original INT8
input scale, and the retained original INT8 traces. It uses all126 consumed
Kristin/Norman presentations. It does not select a new operating point.

The protocol was committed in `02f4e1c` before inference, SHA256
`cb005c82f49cf4cdfc1a205a4959e6a04aee512cb24ac9b6547996e2a7f93312`.
The source verifies the original float archive against `76130cc`, the
historical SavedModel against `ee0becbf`, and all31 parameter fingerprints
against the original checkpoint. Optimizer iteration is70000. Each float arm
is independently reset and repeated exactly on every clip. Every frontend
hash and original INT8 event must match the original126 baseline.

At the frozen threshold `.91796875`, five-score mean and67-step refractory:

| Arm | Hey Nyx positives | Common confuser clips |
| --- | --- | --- |
| Original float | 15/36 | 6/54 |
| Input-rounded original float | 15/36 | 6/54 |
| Original INT8 | 14/36 | 7/54 |

Norman command presentations are0/9 in every arm. Their original-float
source-callback raw peak range is `3.2821578344055524e-08` to
`0.00018187689420301467`; INT8 peaks are allzero. The dominant failure
therefore already exists before INT8. This does not identify its training or
phonetic cause. Quantization still changes scores and two classifications:

- `kristin-1-hey_nyx_command-quiet.wav`: float and rounded float trigger at
  sample44640; original INT8 misses.
- `kristin-1-hey_next-quiet.wav`: both float arms reject; INT8 triggers at
  sample47040.

The trace audit imports no producer helper and performs no inference. It
uses Python `math.fsum` over every five raw values and independently tracks
arming, refractory time and callback sample indices. It checks378 complete
traces, all126 metadata/frontend hashes, retained UINT8 arrays, all events,
reference-window counts, peaks and summary totals. Its separate aggregation
matches the producer. Reset-repeat and checkpoint authentication are producer
execution checks; the trace audit does not independently rerun them.

## Public evidence and reproduction

Release asset: `bryce-v30-original-float-diagnostic-v1.tar.gz` under
`research-hey-nyx-20261110`. The manifest lists every payload hash. Included:
complete three-arm raw traces,126 records, result, independent trace audit,
frozen protocol, original baseline report/UINT8 traces, and exact source.
No new audio, firmware, credentials, UART or device state is included.

From a checkout containing this document, extract the evidence archive into
`EVIDENCE`. Use the pinned packages in its protocol. Recount without TensorFlow:

```sh
python research/wakeword_p4/experiments/audit_bryce_original_float_traces_v1.py \
  --study EVIDENCE/study --baseline EVIDENCE/baseline \
  --protocol EVIDENCE/protocol.json --output NEW-AUDIT.json
```

For the full inference execution, obtain these existing immutable release
assets from the same release; authenticate the listed SHA256 before use:

- `heldout-bryce-development-v1.tar.gz`,18560971bytes,
  `8222367deff176deafb78e925f50c389259b95ca87389bf1e81d06214e459b5f`.
  Use the original extracted `data` directory. Do not synthesize replacements.
- `gpu-v30-original-float-handoff-v1.tar.gz`,570686bytes,
  `6f19e79f63aedde9e8afb0e3229e9f260179099cfaa27590745856233889f83b`.
  Extract its files into `ORIGINAL-MODEL`. The inference command authenticates
  all15 payloads to their immutable git manifest before loading the graph.

```sh
python research/wakeword_p4/experiments/bryce_v30_original_float_diagnostic_v1.py run \
  --archive gpu-v30-original-float-handoff-v1.tar.gz \
  --model-dir ORIGINAL-MODEL --data ORIGINAL-DATA \
  --saved-study EVIDENCE/baseline --protocol EVIDENCE/protocol.json \
  --output NEW-STUDY
```

Keep the repository's exact `02f4e1c` inference/helper sources and required
historical git objects available. The protocol authenticates source hashes
and allpackage versions. Output directories/files must not already exist.

These are consumed synthetic development results at an INT8 operating point,
not an optimized float threshold, independent acceptance, human phonetic
truth or real-user generalization. They do not authorize model promotion,
threshold rescue, changes to current matched training, or final-set access.
