Historical v30 on Kristin / Norman development v1
================================================

The scoring source and protocol were committed/pushed as `a1df1c4` before
inference. Protocol SHA256:
`4490de925b84521c6aef2bd007f1937e250b0748bedb70b01a68e685c6a05a52`.
All126WAVs from the separately published `heldout-bryce-development-v1.tar.gz`
were authenticated before model load. No threshold, audio or model selection
was performed on this cohort. The v30 threshold stays .91796875.

Full-source and generator-prefix-window hits both equal14/36. Common confusers
trigger7/54 (Next6, Mix1, Nick0); opposite Hi ESP phrases trigger0/36.
Bare Hey Nyx is11/18; Hey Nyx followed by check-the-volume is3/18.
Kristin12/18 target and7/27 confuser; Norman2/18 target and0/27 confuser.
All126 reset replays were exact, with no host/firmware-arithmetic event
differences or events during the added application zero prime.

These are historical-reference development diagnostics, not a candidate
acceptance result or comparison with Hi ESP: the latter remains pending its
own frozen official replay. No continuous background is included, so no FAR
or matched-FAR inference is possible. Generator-prefix timing is not verified
acoustic phonetic timing. No physical microphone, playback or board was used.
No final sets were accessed. The saved raw scores cover every clip, including
misses and confusers; do not choose a new threshold to rescue the result.

Reproduce using the source tree from this archive and the data archive:

```sh
python source/research/wakeword_p4/experiments/score_bryce_v30_baseline_v1.py \
  --protocol study/protocol.json \
  --data /path/to/heldout-bryce-development-v1/data \
  --model-dir model --output replayed-v30-study
```

The exact interpreter/frontend package versions are pinned in the protocol.
The source tree retains relative paths needed for helper hash validation.
The source WAV/PCM, model, helper, protocol and output hashes are in the
reports/manifests. `study/scores.npz` retains every UINT8 output keyed by
`output_0`..`output_125`, in the original data-manifest order.
