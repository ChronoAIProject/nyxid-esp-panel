# Training-only frontend/cache factorial feasibility

Prelock3b5dbf54 ran once on RTX4090, actual process session35015 exit0. Strict original Binary40 final20000 weights, no optimizer. All source/checkpoint/script hashes passed. Complete probability/feature arrays and stdout are preserved with terminal hashes.

The fixed first12 unique positive native sources in original training-ledger order all belong to en_GB-aru-medium speaker3. This is a narrow within-voice diagnostic, not balanced crossvoice evidence. One fixed first training negative is used as preceding speech; its duration is padded to480samples, followed by8160zero samples (the proposed8000gap rounded to a complete480block). Equal-length zero prefix is its matched control. Prefix includes81600zeroPCM; total113760samples. Targets are the original native training files, not the rendered augmented training tensors.

Cross frontend suffix from silence/speech history with neural cache from silence/speech prefix at the same complete-call split. Retain the frontend residual frame in each suffix; it is part of frontend history, not silently discarded. Off-diagonal cache/suffix combinations are artificial interventions, not physical deployment resets. Both diagonal split outputs equal unsplit full outputs exactly (maximum probability error0 for all24 checks).

| Frontend history | Neural prefix cache | Target-container five-mean crossings |
|---|---|---:|
| Silence | Silence |12/12|
| Silence | Speech |12/12|
| Speech | Silence |12/12|
| Speech | Speech |12/12|

This verifies the isolation implementation and gives no crossing-level history failure in these12 training native utterances. It does not establish absence of score changes, INT8 agreement, continuous-decoder behavior, crossvoice robustness or a solution for consumed continuous failures. No threshold, frontend, model cache or training recipe is changed. A large blind context-training rerun is not justified. Prior C161 speech-history training already improves short gaps while losing British bare coverage, and must not be repeated under a new name.

Next useful scope is a preregistered training-only source/voice-stratified context diagnostic with fixed source-independent selection and both bare/connected commands, after checking original label references. It should preserve all strata and failures rather than selecting successful sources. Current12 can only be a feasibility gate, not evidence that the full requested model is ready. C161 remains baseline; Mac/P4 claims separate.
