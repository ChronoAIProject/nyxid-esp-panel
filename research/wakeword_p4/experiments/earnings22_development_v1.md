Earnings22 complete continuous development v1
============================================

This is a preparation/provenance handoff, with no wake-detector inference.
The full source-reservation and preparation recipe were committed/pushed at
`ca9c7f5` before downloading/decoding the media. Twelve complete calls were
selected deterministically from the original125-row metadata. Company-country
groups and within-group files were ordered by frozen SHA256 strings; no
listening, model score, audio quality or target-recognition result selected
these calls. Company country is not a speaker-country or accent label.

Source: [revdotcom/speech-datasets Earnings22](https://github.com/revdotcom/speech-datasets/tree/c05ab6fd8b4b627d123c922a22a39e993dd37635/earnings22),
Del Rio et al., *Accents in Speech Recognition through the Lens of a World
Englishes Evaluation Set* (2023), [paper](https://arxiv.org/abs/2203.15591).
The archive preserves original metadata, README, transcripts, LFS pointers,
license text and exact source URLs/hashes. The upstream LICENSE expressly
covers transcripts/associated text under CC BY-SA4.0; do not imply it grants
that same license to audio. This package redistributes no original/decoded
media, native samples or derived frontend arrays. Re-download the pinned
public original media directly from upstream for local preparation.

Use only as new development. These sources are excluded from training and
all existing final sets. No reference was found in the recorded research
source/manifest texts on either branch at reservation time; this is not proof
of speaker/pretraining independence. Call participants may recur; corporate
calls do not represent household acoustics, Mandarin or population-wide FAR.

The source transcripts are screened for exact sequences hey nyx / hey nix /
hey nicks / hi esp / hi e s p. All transcripts and calls remain in the cohort;
a match stops negative readiness rather than selecting an easier replacement.
No match proves only absence in the supplied transcript, whose omissions and
errors remain unknown. No ASR-based source selection or regeneration occurs.

Preparation uses complete gapless MP3 decoding, with bitexact flags, scalar
CPU flags, one decoder thread and no dithering. Native float PCM is separately
streamed into a sample counter/hash; full mono16k PCM16 duration must match
that native decoded duration within one16k sample. This preserves decoded
logical audio, not an unavailable pre-MP3 master. The only declared transforms
are mono downmixing and rational sample-rate conversion. No gain/normalization,
trimming, VAD, denoising, speed/pitch changes, inserted gaps or concatenation.
Original music, pauses and operator announcements remain.

The existing MicroFrontend runs continuously across each entire call, in
10-second I/O blocks, with just one final0..159-sample zero pad. The manifest
separately hashes original LFS media, native decoded floats, full WAV/PCM,
frontend NPY and array bytes, and records all sample counts plus actual runtime
binary identities. This is integrity/geometry evidence, not wake accuracy.

To prepare locally, install the pinned Python numeric/frontend versions in
`reference/protocol.json` and FFmpeg9.0.2, then use:

```sh
python source/research/wakeword_p4/experiments/prepare_earnings22_development_v1.py \
  --protocol reference/protocol.json --sources sources \
  --output /fresh/local/earnings22-prepared
```

The script fetches and validates the exact original media from upstream.
Retain original failures and partial output; it never publishes audio or
selects a successful subset. Decode logs and commands stay with each local
prepared record. Native-duration and source/PCM hashes must match before a
future same-PCM Nyx/officialHiESP comparison. Bitexact flags alone do not prove
cross-platform equality: if a different binary/runtime yields different bytes,
report that difference and freeze a new explicit preparation before scoring.

No model, threshold or event score has been evaluated by this package. A
separate pinned scoring protocol is required for historical baselines and any
candidate that passes its original gate. Never report zero events or a FAR
upper bound merely because preparation passed; there are no detector results.
Nominal Poisson bounds also do not account for clustered/correlated calls.
No final testing, playback, microphone, board/OTA, AEC/AGC/FIFO or production
device operation occurs here.
