# Original paired raw event timing diagnosis

Status: the four original event JSONLs were anonymously downloaded, authenticated
and actually processed on Mac on 2026-11-14. All 552 points, 372 ordered records
per point and every emitted event coordinate passed. No model was invoked and
no audio was accessed. These are retained diagnostic results, not new recognition
or acceptance trials.

`analyze_raw_near_word_events_mac_v1.py` requires all four original files bound
by the retained independent 552-point scoring audit, plus both unchanged
manifests (360 synthetic records and 12 complete background recordings). The
manifests are already present on Mac and their SHA256 matches the execution
lock. No WAV decoding, model invocation, playback or recording is needed.

The reader checks original file hashes against both the independent audit and
grouped report, all 138 thresholds per model, all 372 ordered records per point,
every event step/sample clock including final clamping, and all four generated
support offsets. It compares every complete point summary with the original
published summary and recomputes positive/negative/background counts. It rejects
duplicate JSON keys and missing or displaced original records.

Every literal/carrier group is retained at every threshold, including zero-event
groups. Event and activated-clip counts are split into four intervals: before
source, within generated-prefix support, between prefix and generated-container
ends, and at/after the generated-container end. Exact boundary samples enter
the later interval. First-event signed offsets retain min/median/max. A clip
can contribute to more than one interval if it has multiple events. These are
generated allocation intervals, not phonetic truth, measured wake latency or
evidence that the word was recognized too early.

```sh
python3 research/wakeword_p4/experiments/analyze_raw_near_word_events_mac_v1.py \
  --gpu-ref 4d4de98d4ee3217ec5ee73903cc6da9520e4fb8e \
  --events-dir FOUR_ORIGINAL_JSONLS \
  --synthetic ORIGINAL_360_MANIFEST.json \
  --background ORIGINAL_12_MANIFEST.json \
  --output NEW_REPORT.json
```

Constructed checks exercised six interval boundaries, valid events/final sample
clamping, and six invalid-input rejections (sample clock, signed offset,
duplicate step, out-of-range step, boolean step and duplicate JSON field).
Those constructed checks concern reader behavior, not the original
probability-to-event oracle. The actual run is described below. All previous
failures, working-point requirements, fresh-development and acceptance boundaries
remain.

## Actual intake and full event reduction

The original archive supplied at GPU commit `baa416f3` is 2,028,414 bytes,
SHA256 `b82721e48cff76cdc191c3d12e97eabcfe04eb335b27a44b9f6f6ce92a6cd328`.
The exact six-member set, regular-file types, embedded manifest bytes and every
payload size/SHA were verified before writing controlled member paths. The
attached original audit is byte-identical to commit `1e64ab8d`, and all four
JSONL hashes equal that audit and the original grouped report.

The unchanged, previously published reader actually ran and exited 0. It
checked **205,344 ordered records** across all 552 points, their original
summaries and every event/sample/offset, including background counts. Results:

- `results/mac-original-paired-event-packet-intake-v1.json`: exact intake evidence.
- `results/mac-original-paired-raw-support-timing-v1.json.gz`: complete lossless
  reader output, 297,944 bytes; SHA256
  `90f1b3672dbf803280c018f354f7f521f65e4688b66ba22733373ad5f46f1da8`.
  Decompressed JSON: 18,178,178 bytes, SHA256
  `5c21896c550b0a79e863daae7dffcdc4878fded649cf3f0b6dfe9ac64815cbd3`.
- `results/mac-original-paired-raw-support-regions-v1.json`: every literal/carrier
  group's range over all 138 points and over every FAR-qualified point, including
  groups with zero events. Willow's two empty FAR regions remain empty.

The common-negative errors in both Nyx arms' FAR-qualified regions all occur
after their generated-prefix boundary. That locates them in the retained
generation metadata; it does **not** prove the wake word had finished or that
the following command caused the error.

| Common literal and carrier | Control, all 93 FAR points | Context, all 138 FAR points | Support location |
| --- | ---: | ---: | --- |
| Hey next + set a timer | 6/9 activated | 6/9 activated | Prefix-to-container interval |
| Hey next + turn on the light | 4–5/9 | 3–5/9 | Prefix-to-container interval |
| Hey Nick + turn on the light | 4–5/9 | 0–6/9 | Prefix-to-container interval |
| Bare Hey next | 0–4/9 | 0–2/9 | Control spans later two intervals; context at/after container |
| Bare Hey Muse | 0/9 | 0–1/9 | At/after generated container |

These ranges use different all-FAR regions and are not matched-threshold gains.
The complete 552-point file retains every threshold for such comparisons.
For the timer cohort, the same six input records activate throughout their
arm's entire FAR region: all three variants of each original length-scale
0.9 and 1.0 source; all three variants of the 1.1 source remain unactivated.
Those settings belong to the already-consumed synthesis protocol; this analysis
does not change PCM speed or discard a source. Event-minus-generated-prefix
offsets are +186.5625…+258.1875 ms control and +138.1875…+336.5625 ms context.
One voice and related variants do not establish independent speaker coverage or
a causal duration effect. In particular, don't retune a threshold or train on
these diagnostic failures to claim new independent quality.

The findings rule out the description that all residual common errors are
events before the recorded generated-prefix boundary. They leave representation,
source and carrier interactions unresolved. The separate pretrained-encoder
freeze experiment tests a different mechanism; the timing result does not prove
that freezing will succeed.

Recreate the region report directly from the complete published output:

```sh
python3 research/wakeword_p4/experiments/summarize_raw_support_timing_mac_v1.py \
  --timing research/wakeword_p4/results/mac-original-paired-raw-support-timing-v1.json.gz \
  --output /tmp/mac-original-paired-raw-support-regions-v1.json
```

Use new output paths. The original reader command above can recreate the full
JSON from the authenticated public archive and the two unchanged manifests;
gzip is only storage compression and preserves all output bytes. No inference,
speaker playback, recording, new TTS, P4 flash or fresh data is needed.
