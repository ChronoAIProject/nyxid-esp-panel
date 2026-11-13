# Prospective development background: full-waveform preparation

Freeze before any detector scores. Inherit exact test4/5development and
test6/7acceptance roles from4090-prospective-background-role-plan-v1.json.
This component authorizes lossless preparation only, not detector evaluation.
Acceptance files and all original final sets remain unopened/excluded.

For background FAR, each complete embedded WAV is the replay unit. Decode all
samples, preserve original encoded bytes and PCM SHA, and derive exposure only
from sample_count/16000. No splitting at diarization timestamps, VAD selection,
trimming, concatenation, synthetic padding credited as exposure, resampling or
record dropping. Reset model only between recordings, never between annotated
segments. Count every event under the future fixed event-latch/refractory
policy. No transcript guarantees absence of target phrases: describe this as
unscreened conversational background and retain every event conservatively.

The earlier strict20ms annotation audit remains failed in8/42recordings. That
audit diagnoses metadata consistency; it does not establish missing waveform
samples. The new preparation rule verifies actual full-container sample counts
and encoded/PCM hashes, keeping every invalid annotation interval visible.
It neither changes annotation tolerance nor repairs the original metadata.
Diarization boundaries are not consulted for exposure or waveform selection.
This distinction is fixed without any detector score knowledge.

Require source file SHA/size, mono PCM16/16k, full decoded sample count equal
the independent diagnostic ledger, original encoded/PCM SHA, all42planned
recordings, no within-pack duplicates and >=6hours. Known dev0..3duplicates
remain forbidden. Person-level independence, unknown original mirror editing
and cross-all-training/unopenedfinal overlap remain unproven and must not be
claimed. A package passing these checks is integrity-prepared, not fully
qualified independent development or acceptance.

Output manifest explicitly sets scoring_authorized=false and
background_qualified=false with prospective-development-negative-unscored
record roles. Before scores, complete positive/confusion source reservation,
training/final isolation audit, mechanical synthesis gates, fixed threshold
grid/tie rules, v30/officialHiESP same-PCM comparison and independent acceptance
protocol. New metadata limitations never authorize automatic promotion or
post-score selection, extra exposure or changed gates.
