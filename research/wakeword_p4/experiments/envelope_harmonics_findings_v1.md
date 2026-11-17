# Envelope perturbation: harmonic-estimator review and bounded next intervention

Prelock c1825c5f, actual exec44083 exit0. All48 original bare training sources plus48 original same-reference negative sentences, two fixed transforms each,192 records. All96 source size/SHA identities verified, original statuses unchanged;25 bare quarantines remain quarantined. All288 original/transformed F0-estimator tracks preserved with archive SHA. CPU2 preprocessing, no neural inference, synthesis or training.

| Cohort / alpha | Original selected frames | Still voiced by estimator | Median absolute cents | p95 absolute cents | Comparable frames >600 cents |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negative .9 |2034|2020|1.2324|13.1301|8|
| Negative1.1 |2034|1991|1.1398|12.2863|7|
| Bare .9 |1274|1262|1.3790|17.1831|7|
| Bare1.1 |1274|1253|1.2163|15.8836|0|

Comparable excludes transformed tracks with no local peak; all original-selected frames remain in the voicing-retention denominator. Autocorrelation is not measured F0 truth, and correlated frames are not independent samples. The small estimator changes support a different mechanism from the failed whole-spectrum pitch perturbation, but do not prove phonetic preservation, unchanged true F0, or recognition benefit. No acceptance gate was chosen from these observations.

The48 negatives are “What's next?”, “Please mix the ingredients.”, “Turn on the lights.” and “Set a timer.”,12each,4references. They are not five-phrase bare lexical contrasts. They cannot establish short-word discrimination or justify applying a transform only to positives.

A defensible next controlled intervention is now bounded: apply the fixed envelope transform to the existing C161 broad training bare target AND same-source Nick/Next/Mix/Muse inventory, not the quarantined pilot sources. Reuse the authenticated31644-row inventory exercised in the pitch experiment. Choose one .9/1.1 value per source-pair/variant hash, independent of phrase/class; preserve all source/label/duration identities and full continuous frontend state. Select original or derivative at a fixed50% using separate RNG, preserving the original C161 batch schedule, masks, graph, losses, initialization and final-only20000 budget. This is a different transformation at the same augmentation budget, not pitch magnitude rescue. No new loss or development voices.

Before training, construct at most2GiB derived feature cache with sequential CPU2 preprocessing and >=100GiB reserve; preserve full per-row source/PCM/feature hashes and all mechanical failures. Verify paired actual batches and exact unchanged input/mask slots. A later training launch requires its own immutable recipe and all cache checks. Evaluate same fixed deployed profile against saved C161 with normal hits lost, crossvoice/commands, daily conflicts, background and constructed-word diagnostics jointly. No threshold/checkpoint/calibration rescue. The hypothesis is reduced dependence on spectral-envelope coloration; it is not established as the cause of Joe/Mac failure. A null or harmful candidate stays rejected.
