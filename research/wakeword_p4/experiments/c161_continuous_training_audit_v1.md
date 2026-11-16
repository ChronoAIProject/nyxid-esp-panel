# Continuous-history training intervention preparation

C161 remains the baseline; bc01 remains experimental. No candidate is promoted by this data preparation.

## Source audit

`cache_chatterbox_training_v1.py` creates each bc01 expansion row from a fresh `features()` call over 81600 zeros + one admitted waveform + 8000 zeros. All 84 rows therefore omit preceding speech. `causal_chatterbox_batch_v1.py` replaces four positive and four negative slots per batch with those rows. The parent `causal_sequence_batch_v1.py` draws separate ambient negative sequences; this does not place ambient or speech before the positive input. Original synthetic material also uses individually rendered waveforms (`build_causal_onset_material_v1.py`); broad rendering pads individual speech with a lead and may add noise/RIR (`render_hey_broad_training_v1.py`). This is source inspection of the named paths, not proof that every historical training example has no acoustic adaptation or that the mismatch alone causes observed losses.

The paired feature intervention evidence in `c161_continuous_chatterbox_findings_v1.md` motivates a training-only continuous-history experiment. Evaluated builtin development voice/audio remains excluded.

## Prepared paired material

Preregistered commit 10d649e3 was pushed before extraction. Extraction completed exit 0. All 84 previously admitted training-only Chatterbox targets are retained, with two fixed gaps (0.5 and 2 seconds). A deterministic index rule chooses another admitted daily-negative utterance of the same synthetic voice as prefix; no model-score selection. One condition uses that prefix audio, the other equal-duration zeros. Both use the exact target PCM, identical onset phase and prefix-end supervision.

Frontend extraction runs on the full prime/prefix/gap/target/tail waveform before cropping the feature history on a multiple-of-three frame boundary. Each retained sequence includes at least 360 feature frames before target onset, entire target and 8000-sample tail. This explicitly changes full-wave training to a history crop; the model starts at that crop, while the frontend retains earlier acoustic adaptation. The >=309-frame context margin precedes positive supervision. The silent control has identical geometry, making this a paired intervention rather than a direct bc01-only comparison. Negative prefix supervision is safe only to the extent of the admitted synthetic daily-negative labels; independent ASR previously matched all 48 negatives, not human phonetic ground truth.

168 pairs (72 positive,96 negative), 95386 frames per condition; each feature binary 7630880 bytes. All168 target regions differ between frontend conditions. Per-row feature hashes and masks were recomputed from the produced binaries. Whole target bytes and paired geometry were checked during extraction; original model inference was not invoked.

## Sampler check

`causal_chatterbox_context_batch_v1.py` preserves the bc01 parent source draws, replacing its eight Chatterbox slots with the selected target's paired cache row. A separate seed730 RNG chooses the gap variant identically in both arms. Twelve checked batches preserve all120 other input tensors/descriptors, retain32 positive presentations total, and use four contextual positives/four contextual negatives. Speech/silence arms have identical masks, source indices and chosen pair IDs; only feature values/condition metadata differ. These are sampler mechanics, not recognition evidence.

## Next training protocol requirements

Use two sequential RTX4090 arms, original shared initialization, seed726, fresh Adam1e-5, frozen BN and20000 final updates each, matching bc01 except this declared paired cache. Lock hashes/runtime and source identities before execution. Export only final checkpoint once per arm, retain failures, no checkpoint/threshold rescue. Report both arms on the consumed fixed-profile diagnostics including continuous sessions, British bare regression and daily common-language conflicts. Preserve C161 baseline regardless of intervention outcome. No protected acceptance access, new independent-voice claim, P4 qualification or retained-command PCM claim.
