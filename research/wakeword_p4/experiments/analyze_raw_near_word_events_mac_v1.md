# Original paired raw event timing diagnosis

Status: reader prepared; the four original event JSONLs have not been received
or processed on Mac. No new recognition result is implied by this document.

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
This checks reader behavior only, not actual paired-raw recognition or the
original probability-to-event oracle. An actual report will be published only
after the authenticated original files are processed. All previous failures,
working-point requirements, fresh-development and acceptance boundaries remain.
