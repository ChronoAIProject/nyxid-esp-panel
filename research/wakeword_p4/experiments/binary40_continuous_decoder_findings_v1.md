# Continuous misses are absent crossings, not decoder blocking

All96 target containers across four models and two gap settings were inspected in the original saved continuous UINT8 traces. Trace SHA256 was authenticated; the fixed decoder was independently reproduced across the entire stream, including prime and all boundaries. Emitted steps match both original reports and the shared event implementation exactly. No model inference, optimization or policy change occurred.

| Model | Gap | Emitted /12 | No five-code crossing /12 | Miss with blocked crossing /12 |
|---|---|---:|---:|---:|
| C161 |0.5s|0|12|0|
| C161 |2s|2|10|0|
| Original Binary40 |0.5s|7|5|0|
| Original Binary40 |2s|8|4|0|
| Continued64 |0.5s|2|10|0|
| Continued64 |2s|7|5|0|
| Wide128 |0.5s|2|10|0|
| Wide128 |2s|6|6|0|

There are62 missed containers; all62 lack a sum5>=1175 crossing inside the whole clip container. This rules out refractory/rearm blocking as the immediate explanation for these particular misses under the fixed protocol. Blocked high calls after an emitted event are retained in the per-container evidence but do not represent missed containers. No threshold/rearm/refractory rescue is justified. A lower threshold would be a policy change with different false-positive behavior, not evidence that current misses were blocked.

The preceding temporal-mask audit found absent post-positive negative supervision, but these continuous traces do not support its suggested decoder-rearm mechanism. Do not launch a rearm-targeted training intervention on that basis. This also does not prove temporal supervision is optimal: frontend history, neural context and score margins remain distinct possibilities.

Next evidence should separate frontend-history effects from neural-context effects using original training-only utterances in a fixed continuous context diagnostic, before choosing any training recipe. Prior C161 silence/speech context-pair training already exists; a new context recipe must be distinguished from it and its failures. Do not use consumed evaluation failures as training samples or select a threshold from the saved maxima. Container clocks are not phonetic endpoints or first-command retention evidence, and no P4/general-human claim is made.

Reproduce with `/usr/bin/python3 research/wakeword_p4/experiments/audit_binary40_continuous_decoder_v1.py`. Full original trace/report hashes, per-container maximum sums, crossing counts and blocking counts are preserved in `4090-binary40-continuous-decoder-audit-v1.json`. C161 remains the baseline.
