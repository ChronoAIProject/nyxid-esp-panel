# Independent audit of saved GPU Joe/background evidence

`audit_joe_background_saved_scores_v1.py` authenticates three original release
archives against immutable git manifests and independently recounts the saved
scores and UART. It performs no model inference or QEMU execution.

The archives and all payload hashes match: Joe fixed baselines2526 payloads,
background256, Joe float diagnostic7. For all360 Joe inputs, the full original
WAV PCM matches `input.pcm` after its16-byte transport header; PCM SHA256 and
CRC32 are recomputed. For all402 saved official runs, UART parameter/event/
completion lines agree with reports,512-sample call counts, stage104, getter
and frozen metadata. Every original v30 score trace is recounted with separate
latch/refractory code. The Joe float means are independently recomputed from
raw float values with `math.fsum`; original INT8 arrays match the baseline.

| Joe fixed baseline | Positive | Bare | Command | Common | Cross |
| --- | --- | --- | --- | --- | --- |
| v30 | 0/72 | 0/36 | 0/36 | 0/216 | 0/72 |
| Hi ESP | 64/72 | 29/36 | 35/36 | 18/216 | 0/72 |

All18 official common activations have intended text `Hi E S B.`. On Joe,
original float, input-rounded float and INT8 all have0/72 positive triggers at
the frozen point. Their maximum positive five-score means are respectively
`.8511093139648439`, `.8495105981826783` and `.8054687500000001`.

Background totals455204864 original samples,7.902862222222223 hours. Both saved
engines have0 events. Independent `-log(.05)/hours` gives the nominal one-sided
Poisson95% upper bound `.37906927760049125` per hour, agreeing numerically with
the producer's chi-square calculation. This does not remove clustering.

Reproduction requires the three existing release archives named in the audit
report. Place each unmodified archive beside its extracted contents:

```text
EVIDENCE/mac-joe-score-receipt-v1/<archive>, extracted/{manifest.json,replay,tools}
EVIDENCE/mac-background-score-receipt-v1/<archive>, extracted/{manifest.json,replay,tools}
EVIDENCE/mac-joe-float-score-receipt-v1/<archive>, extracted/{manifest.json,diagnostic,tools}
```

Use the original Joe render directory from its already authenticated public
release. Keep git objects `90bca76`, `0b8286b` and `73f4562` available:

```sh
python research/wakeword_p4/experiments/audit_joe_background_saved_scores_v1.py \
  --root EVIDENCE --render ORIGINAL-JOE-RENDER --output NEW-AUDIT.json
```

The background freeze retains its Joe parent under `files.development_manifest`;
its actual background input is separately pinned under `manifest`. The audit
uses that background pin and compares all42 original records. No execution
pins or producer files were modified.

The background archive explicitly omits original WAV/`input.pcm` bytes. This
audit authenticates the published manifest, saved scores and UART fields; it
does not independently decode or authenticate the excluded background audio.
It also does not independently authenticate actual GPU engine binaries or
rerun reset-repeat behavior. Joe timestamps remain generated-container/RIR
references, not human wake-prefix boundaries. These consumed development
results do not promote a model or establish independent/hardware acceptance.
