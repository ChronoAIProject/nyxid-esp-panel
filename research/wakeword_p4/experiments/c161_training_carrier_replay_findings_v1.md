# Admitted training carrier fixed-profile replay

Preregistered 5ec07f2e. All 84 admitted original training waveforms were replayed on C161 and both final paired INT8 models, all 252 runs completed with genuine process exit0. Single-thread BUILTIN_REF deployment simulation, no CPU training. Original fixed .91796875/5/67/half-threshold event policy, continuous 81600-zero prime, no boundary reset, native480 return clocks. No quarantined audio, acceptance access, generation or tuning.

| Training fit diagnostic | C161 | Silence-history arm | Speech-history arm |
|---|---:|---:|---:|
| Bare | 1/6 | 6/6 | 4/6 |
| Command | 6/30 | 29/30 | 26/30 |
| Daily negative events | 0/48 | 0/48 | 0/48 |
| Prime events | 0 | 0 | 0 |

All admitted bare examples can activate under the silence-history arm despite their small number. Therefore carrier scarcity alone does not explain the speech-history arm's isolated-input failures. Speech arm bare misses occur in isabella and fable; command misses are two onyx and two isabella. Silence arm misses one onyx command. These are training-fit observations on original full waveforms, not independent development or cause localization.

The speech-history arm's already published continuous 0.5s gain (12/12 vs silence4/12) coexists with isolated training-fit loss. Training cache history crops and frontend context differ from this full-wave replay; model startup/history mismatch remains a plausible contributor. Do not conclude that speech history should be removed, that oversampling bare will fix cross-voice failures, or that high training fit qualifies the silence arm.

Next useful controlled design should retain both isolated and continuous history support with identical total Chatterbox presentation budget, rather than replace all such slots by a single context condition. Lock paired sampling and masks and preserve base sources; compare against both existing arms. This is a new mixed-context hypothesis, not a guaranteed repair, and must still report British bare, Joe/Daniel and daily lexical-conflict regressions on consumed diagnostics. Future coverage requires training-only diversity and independent evaluation; existing development identities remain excluded.

All252 raw UINT8 traces, per-input reports and original aggregate report are archived; every archive member's size/SHA verified. Automatic signed prefix timing is descriptive, not human phonetic latency or first-command retention. C161 stays baseline; no candidate promotion.
