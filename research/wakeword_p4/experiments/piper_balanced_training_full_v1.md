# Full balanced inherited-training conditioning-ID synthesis v1

Prerequisite: c683002 process-only32-ID pilot384/384 original mechanical checks
and independent artifact/conversion/duration audit pass. This demonstrates
fixed-host process feasibility only, not model or acoustic quality acceptance.
Motivation and source/identity limits remain those in the frozen pilot plan.

Freeze before inference: ALL672 IDs in original order from
numpy default_rng(726).permutation(904)[:672], matching the authenticated
4090-piper-training-phrase-coverage-v1.json training_ids list. Each ID is
training-only already; no reserved waveform/replay manifest/feature/score read.
All672 ×3length scales(.9,1,1.1) ×4texts(Hey Nicks.,Hey Nick.,Hey Next.,
Hey Mix.) ×1draw=8064 attempts, exactly3positive+9negative perID. Original
pilot outputs remain untouched and are NOT added to/reused in this new corpus;
render all8064 once,torchseed197726,fixedID/speed/phrase order,batch1 on4090.
This is not672 independent physical speakers: conditioning IDs and shared
LibriTTS checkpoint ancestry are explicit. No development failure-selected IDs.

Same existing e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6
model,119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848 config,
piper sample generator b5b36e266576693ea47db129fbfe6bbc06164762 and exact
actual local source hashes/modifications recorded. Same-ID pairs,slerp.5,
noise/noise_w .333 each,no max-length truncation. Preserve per-token duration
arrays/inputphones/tokens,whole nativefloat32/resampledfloat32/PCM16WAV,
source/run/script/package/grid/order/seed/file/PCM hashes. No normalization,
trim,retry,endpoint repair,failed-row replacement or wake-model score selection.
Durations are exposed generation process values,not acoustic phonetic truth.
No claim of CUDA generation bit replay or newly fetched upstream byte proof.

Apply unchanged pilot gates to every attempt:finite mono nonempty/nonzero,
16kPCM16,.2..5s,RMS>=.001,native/resampled abs<1,no PCM endpoints,whole lengths,
native length equals finite nonnegative exposed duration sum,ceil resample
length. Preserve all8064 attempts and all failed checks; no filtering or
threshold/tolerance relaxation. Independently recompute all file/PCM hashes,
source/grid/allowedID/token ledger,duration arrays/length sums,exact
resample_poly320/441 and pinnedlibsndfile conversion. Sharedlibrary numerical
independence limit remains. Originalrun/report/manifest must be preserved.

Only generation and read-only audit authorized by this plan. No model scoring,
features,training,threshold choice,old-source quality repair or promotion.
If full corpus passes, separately freeze complete-source feature/augmentation
plan and sampler design; otherwise retain/report failures before any dataset
use decision. Do not silently certify inherited training data. All original
fixedoldstage and acceptance gates unchanged,v30/newdev/finals retained.
