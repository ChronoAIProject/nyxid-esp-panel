# Historical v30 miss attribution on consumed development

The retained 126-stream UINT8 output archive supports a direct distinction
between a weak classifier output and an event-policy miss. This post-hoc
diagnostic loads only the original report, scores and source manifest. It does
not load audio or a model, change thresholds, evaluate another detector policy
on background, or consume any new/final data.

Every original event sequence is reproduced with integer sums: raw threshold
235/256, five-score sum threshold 1175, half-threshold rearming and 67-step
refractory. Direct sliding additions also agree with every convolution sum.
All 126 event lists match the saved firmware/host report exactly; prime events
remain zero. Source offsets retain the original callback/block clock.

| Voice / phrase | Original events | Single-score maximum reaches original threshold | Five-score maximum reaches original threshold |
| --- | ---: | ---: | ---: |
| Kristin, bare | 9/9 | 9/9 | 9/9 |
| Kristin, command | 3/9 | 5/9 | 3/9 |
| Norman, bare | 2/9 | 2/9 | 2/9 |
| Norman, command | 0/9 | 0/9 | 0/9 |

Of 22 missed Hey Nyx clips, 20 never have a single output reaching the original
threshold. Two have a high individual score but no qualifying five-score mean:
`kristin-0-hey_nyx_command-combined15.wav` and
`kristin-1-hey_nyx_command-quiet.wav`. No clip has a qualifying mean but is
blocked by the latch/refractory. All source-callback UINT8 outputs in the nine
Norman command renders are zero, which is a quantization-floor observation;
it does not prove zero underlying float probability or an audio capture defect.

The 54 common-confuser clips have seven threshold-reaching maxima and seven
original event clips. The 36 opposite Hi ESP clips have neither. Those counts
do not establish a safe unsmoothed policy: continuous-background events were
not recomputed for one-score triggering, and no policy change is proposed.

Native generation also differs between bare and command speech. Kristin's
prefix duration changes from approximately 743–778 ms bare to 522–569 ms with
command; Norman's changes from 650–801 ms bare to 1010–1265 ms with command.
These are generator token durations, not certified acoustic endpoints. Different
native generation/acoustics prevent treating this as a controlled intervention
that merely appended an identical suffix. The three render conditions share
each native source and are not independent speakers.

The useful conclusion is limited: most observed misses already occur before
the final event policy can help at the original threshold. Do not shorten
firmware waiting, discard the five-score policy or lower thresholds on the
basis of this report. It does not isolate a causal training failure or justify
changing the running fixed/per_draw experiment. Keep source/model coverage as
an explicit question for the next training decision, and compare actual Hi ESP
results on the same complete PCM when available.

The raw input scores remain in the existing
`bryce-v30-historical-baseline-v1.tar.gz` release asset, SHA256
`ca84ea47f0ceccc660f58eb4235fef7f18f87ec77e281fc3a48eec751fba51e9`.
The original source manifest is in `heldout-bryce-development-v1.tar.gz`.
Reproduce using NumPy, without inference dependencies:

```sh
python research/wakeword_p4/experiments/diagnose_bryce_v30_saved_scores_v1.py \
  --study /path/to/bryce-v30-study \
  --source-manifest /path/to/heldout-bryce/data/manifest.json \
  --output /fresh/path/miss-diagnostic.json
```

Full per-source values and input hashes are retained in
`results/mac-bryce-v30-saved-score-miss-diagnostic-v1.json`. This is development
diagnosis, not new model quality, Hi ESP parity, hardware or long-call evidence.
