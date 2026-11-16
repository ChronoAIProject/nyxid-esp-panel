# Chatterbox source expansion candidate bc01

Candidate bc01 merits experimental Mac/P4 comparison; retain C161 as baseline.
The new source improves several consumed synthetic voices without recovery on
Joe. Constructed-nearword activations also increase; these are diagnostics,
not a daily false activation rate or an automatic rejection criterion.

| Fixed .91796875, smoothing5 | C161 | bc01 |
|---|---:|---:|
| Legacy target / command | 35/36, 26/27 | 36/36, 27/27 |
| Legacy constructed negatives | 4/288 | 6/288 |
| Jenny target / command | 33/36, 24/27 | 34/36, 25/27 |
| Jenny constructed negatives | 44/324 | 56/324 |
| Joe target | 0/72 | 0/72 |
| Chatterbox builtin target | 5/12 | 10/12 |
| Chatterbox daily negatives | 0/12 | 0/12 |
| Mac Daniel/Samantha Nyx | 1/4 | 2/4 |
| Mac other-target clips | 0/8 | 0/8 |

Paired evidence: Legacy retains35 targets, gains1, loses0; Jenny retains33,
gains1, loses0; builtin retains5, gains5, loses0; Mac Nyx retains1, gains1,
loses0. Legacy false clips retain2/remove2/add4. Jenny false clips retain43/
remove1/add13, candidate Next15/Mix28/Muse13. No threshold tuning.
Two reused background streams remain zero events over10.483944826h and
9.614419583h; do not add their hours or call them fresh exposure.
Jenny8 events precede the synthetic prefix reference; this is descriptive,
not phonetic truth or an automatic failure gate. Full-container hits cannot
prove timely wake-prefix recognition or first-command retention.

Training used the same original initialization, seed726, fresh Adam1e-5,
frozen BN and20000 final updates as C161, replacing four source-matched
positive/negative pairs per128 batch. All other120 tensors preserved in12
paired batch checks. Four reused Kokoro training identities conditioned a new
pinned Chatterbox engine;36 transcript-consistent aligned positives and48
daily negatives admitted,12 ambiguous positives quarantined. Evaluated builtin
development conditioning was excluded from training. Conditioning identities
and random seeds do not certify independent human speakers or pretraining.

Training exit0 in1527.838531s; single final INT8 export exit0, static/runtime
ABI checks pass. Model60952bytes SHA256
`bc011e24ff04f70c3911d39b6cc027a016d68c6fd793124d777ebf5f29edefa8`.
All Legacy372/Jenny408/Joe360, Mac12 and builtin36 replays completed through
the sequential continuation with genuine exit0. Continuous81600zero samples,
no real-audio boundary reset, original event profile. All original report bytes
and1140 UINT8 traces published losslessly; small48 replays separately archived.
Original C161 and frozen v30 remain unchanged.

Experimental handoff: `4090-bc01-mac-experimental-model-v1.tflite.gz` and
`4090-bc01-mac-experimental-handoff-v1.json`. Mac should generate numerical
PCM tests, verify P4 output and whole-device resources/stability, then compare
C161 and bc01 in common speech, accents, connected commands and first-command
retention. No automatic firmware switch or baseline promotion. Hey Willow
remains the primary reference; its same-source builtin target control12/12
is useful but not a universal P4 or human comparison. Practical P4 evidence
and broader daily interference remain outstanding.
