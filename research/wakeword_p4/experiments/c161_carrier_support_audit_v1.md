# Chatterbox expansion carrier support audit

Metadata-only audit verified size/SHA256 of all 96 original training waveforms and their unique indices. No generation, model replay, training or acceptance access. Admission and quarantine remain unchanged.

| Training conditioning identity | Admitted bare | Admitted command | Admitted daily negative | Quarantined bare / command |
|---|---:|---:|---:|---:|
| af_bella | 0 | 8 | 12 | 3 / 1 |
| am_onyx | 1 | 6 | 12 | 2 / 3 |
| bf_isabella | 3 | 9 | 12 | 0 / 0 |
| bm_fable | 2 | 7 | 12 | 1 / 2 |

Only 6/36 admitted positives are bare (30 command); 6/12 requested bare examples were quarantined, versus 6/36 requested command examples. This is a support asymmetry in the new Chatterbox expansion, not the full parent sampler. Three seeds do not constitute three independent speakers. The four references reuse Kokoro training identities; they do not establish broad independent speaker coverage.

This constrains the next intervention: do not call the previous source expansion balanced bare/command coverage, infer that missing bare examples caused British/Joe/Daniel failures, or reverse quarantine based on wake scores. A useful next experiment should first audit source-specific training-only bare recognition and event timing, then test explicit carrier-balanced sampling with an otherwise matched control. Repeating the speech-history recipe alone is insufficient. Report duplicate exposure if the small admitted bare set is oversampled; do not describe oversampling as new voice diversity. No consumed development identity is admitted and no threshold/calibration rescue is proposed.

The daily negative expansion contains only four sentence templates (What's next; mix ingredients; turn on lights; set timer), each with 4 identities x3 seeds. Thus 48 clips do not represent 48 lexical contexts. A future independently specified everyday-language training pack should cover ordinary number/address/action contexts with the same training-only identities, while keeping daily development audio out of training. Existing six-cups development evidence must remain consumed; a pack designed after that observation is a targeted intervention, not independent confirmation.

C161 remains baseline. Paired final models remain experimental. P4 resources, stability and first-command PCM retention are still unverified.
