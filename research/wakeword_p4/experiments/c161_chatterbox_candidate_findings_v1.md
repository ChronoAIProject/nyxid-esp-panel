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

## Subsequent consumed Melo check

The complete474-input Melo replay also exited0:432 synthetic presentations
and42 reused whole backgrounds. Fixed .91796875/5 only, no grid search.
Saved C161 trace identities and per-source hashes were verified before paired
comparison. EN-BR denotes British, not Brazilian.

| Melo target | C161 | bc01 |
|---|---:|---:|
| EN-US bare | 9/9 | 9/9 |
| EN-US commands | 24/27 | 23/27 |
| EN-BR bare | 7/9 | 3/9 |
| EN-BR commands | 14/27 | 18/27 |

This extends the earlier four-cohort improvement evidence with real regressions:
US commands gain1/lose2; British bare gains0/loses4; British commands gain6/
lose2. Total target54/72 to53/72. Do not describe bc01 as uniformly improving
voice coverage. British bare loss is a material reason to retain C161 and make
bc01 experimental only. It must be included in Mac/P4 comparison. No threshold
or training rescue follows from this consumed set.

Both models have zero events on the reused7.902862222h backgrounds; these
hours are not fresh or additive. bc01 also activates7 British reciprocal
official-target clips versus C161 zero. Constructed nearwords remain diagnostic;
whole-container activations still do not establish phonetic correctness or
daily occurrence rates. Complete474 traces, records and result are archived in
`4090-bc01-melo-evidence-v1.tar.gz`, with verified member hashes in its manifest.

## Common-language expansion and six-cups review

Locked32 intended daily sentences x3 seeds, one consumed builtin development
voice, generated96 full watermarked clips (165seconds) and replayed both
fixed models. All tasks exited0. Conversation24/household24/name-address24:
both zero activated clips. Next/mix/six daily sentences24: C161 zero, bc01 one.
These are correlated synthetic clips, not exposure frequencies or FAR hours.

The triggered request is “We need six cups of water.” seed726, index69.
Independent pinned small.en CUDA float16 beam3/no prompt/VAD transcribes all
three sentence seeds exactly as requested. bc01 triggers at0.75seconds for
seed726; the other seeds do not trigger. C161 never triggers on these three.
Descriptive maximum5-step means: C161 .87109375/.6046875/.22734375 versus
bc01 .94921875/.60546875/.1390625. Threshold remains .91796875; no rescue.
The event overlaps ASR's automatic “six” interval .46–.76seconds. This is a
daily lexical-conflict lead, not precise phonetic truth or causal localization.
One successful ASR transcript does not substitute for acoustic/P4 review.

First ASR review failed while serializing already-recursive word dictionaries,
before writing a result. Failure log retained; corrected execution lockv2 and
unchanged audio/ASR parameters completed with exit0. Raw word/timestamp reports
and saved-trace peak checks are published. No new generation/retries to find
cleaner examples, no samples relabeled or admitted to training. Mac should
include natural “need six” contexts in C161/bc01 practical comparison; retain
C161 as baseline, alongside the British standalone regression noted above.
