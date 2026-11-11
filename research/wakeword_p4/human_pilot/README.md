# Paired human wake pilot

Run the recorder with the same Python environment as the wake research
tools (NumPy and SciPy). The server binds to this Mac and saves explicitly
submitted recordings in a private directory outside the checkout.

```sh
python research/wakeword_p4/human_pilot/serve.py --output /path/to/private/captures
```

Open http://127.0.0.1:8796/ in Chrome. Use a pseudonymous participant
code, pick a distance, and read all eight prompts as displayed. The
four pairs counterbalance phrase order; two pairs include an immediate
request. Say “Hi E S P” as separate letters. Use the same microphone
and position for the whole round. Keep misreads in the session so a
reviewer can record them. The input stops after each four-second trial.
If repeating at the second distance, keep the same participant code.
The browser captures mono PCM with browser echo cancellation, gain, and
noise suppression disabled; it may downmix a two-channel input. This
records via the Mac microphone, not the development board.

The UI only saves after the participant selects **Save recordings**.
There is no remote upload, automatic playback, or access to previously
saved audio through the recorder. Keep capture JSON, review HTML, audio
and participant codes out of git and release assets.

## Independent pronunciation and endpoint review

Reuse the GPU team's frozen listening page and annotation validator:

```sh
git fetch origin research-hye-nyx-voice-gpu
git show 3b52402:research/wakeword_p4/training/blind_endpoint_review.html > /path/to/private/template.html
git show 3b52402:research/wakeword_p4/training/collect_review_annotations.py > /path/to/private/collect_review_annotations.py
python research/wakeword_p4/human_pilot/prepare_review.py \
  --session /path/to/private/capture.json \
  --review-template /path/to/private/template.html --output /path/to/private/review
```

Repeat `--session` for distinct participant/condition pairs. The preparer
checks all eight prompts, rate, length and session identity, preserves the
captured PCM, and writes an exact four-second 16 kHz replay copy. The
sample-rate conversion keeps the same duration; the original PCM remains
available. Automation exports require the explicit test-fixture switch and
are marked as such; never score them.

Reviewers open the generated private `index.html` when they can listen
at normal speed. Each records pronunciation match, mismatch or uncertainty
and, for matching clips, a lower and upper bound for the *wake phrase*
endpoint, excluding any request. Use separate reviewers when possible.
The page shows neither detector scores nor automatic boundaries. Export the
annotations and check them with the existing collector:

```sh
python /path/to/private/collect_review_annotations.py \
  --html /path/to/private/review/index.html --input-audit /path/to/private/review/audit.json \
  --annotations /path/to/private/annotation.json --output /path/to/private/review-summary.json
```

Keep misses, misreads, disagreements and incomplete reviews. Label reports
by independent participant and condition, not by total takes. For a
detector event at sample `t`, an independently reviewed endpoint interval
`[lo, hi]` gives delay interval `[t-hi, t-lo]`. A negative or overlapping
interval is not a reliable post-end hit. Use the same frozen source
recordings and engine settings in any replay. The pilot can test the
measurement process; one voice, Mac replay, or HTTP status polling cannot
establish population recall, sample-accurate board latency or call stability.
The separate development-board comparison still needs matched room
conditions, continuous negatives and the full voice path.
