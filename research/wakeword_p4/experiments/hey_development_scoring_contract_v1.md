# Paired development scoring contract v1

This resolves metric/time conventions before any fresh development inference.
It supplements the paired pilot without changing its138threshold grid, gates,
selection rank, model identities, source reservations or acceptance separation.

Custom models: frozen paired Willow/Nyx exports and frozen original v30. Use
raw UINT8 reference-interpreter outputs with uint8-sum-binary64-v1, five-byte
integer sum divided by1280,67step refractory,half-threshold rearm,no epsilon.
Use fresh frontend/interpreter at whole clip/call start only; no internal reset,
VAD, segmentation, added gaps, respeeding or gain. Final frontend padding only.
Authenticate full WAV, decoded PCM and any precomputed frontend before reuse.
Preserve full ordered raw output and exact reset-repeat result for every input.

A one-based raw output step s has frame support end (3*s+2)*160samples from
input start (25ms window,10ms hop,first output after three available frames).
Clamp reported final support to original sample length; keep raw step index.
Event time is input-sample support, not wall-clock inference execution time.

Each target has36positives:4carriers x3speeds x3conditions. A complete-phrase
hit requires at least one event at/after generated_prefix_end_sample and before
or at original container end including saved trailing silence. Compute the
post-full-RIR-prefix hit separately. Also report any-after-source-start hit;
it cannot substitute for complete-phrase recall in the eligibility gate.
Pre-source events (<lead_samples) are the pilot early-event failure gate.
Events during generated prefix are separate diagnostics, never silently removed
from event records. Internal VITS allocation is a generation reference, not
acoustic/human phonetic truth; signed delays may be negative. Preserve every
signed event offset to dry/full-RIR prefix and command-container endpoints.

Macro recall is equal mean over the3condition recalls. Standalone and command
recalls use9 and27positive clips respectively. Eligible iff macro>=.8,
standalone>=.75,command>=.75,no pre-source events,no common/cross events and
nominal one-sided95Poisson FAR upper<=.5/h. FAR upper is
.5*chi2.ppf(.95,2*(events+1))/full_original_hours. All12continuous calls count
with full original samples in exposure; no quiet/talking filtering. Count all
background events, including those whose support is in the final partial hop.
Poisson upper is nominal and does not model event clustering.

Common negatives are the8other literal phrases (288clips), including Hi E S P;
reciprocal target is36cross clips. v30 uses Nyx positives. Official HiESP uses
its own36Hi E S P positives, with Willow/Nyx as cross targets and remaining
7common phrases; report explicit denominators, never relabel detector targets.

Select eligible point by macro,then smaller FAR upper,then higher threshold;
null if none. Save all138eligible/ineligible points and per-condition/speed/
carrier/voice records. v30's original .91796875 point is a separate fixed
reference even if absent from grid. Full curves are descriptive development
results; only the registered ranking selects each paired candidate point.

Official reference retains its13DEFAULT/SET configurations and actual getter
bits,63separate zero-prime returns,512sample input calls,no custom event policy,
all original detect values and complete ordered events. Use the same saved
PCM and generated boundaries. Classic QEMU runtime identity and its bounded
consumed-speech/long-stream certification must accompany results; do not claim
universal P4 equivalence. Prime-after-getter observation gap remains disclosed
until separately instrumented and certified. Official results are diagnostic;
no numeric threshold equivalence with custom models is assumed.

No inference is authorized by this document alone. Execution pins must bind
material/history/timing/candidate identities, scorer/calibrator sources,
runtime/frontend/dependency identities and complete continuous-background
inputs before any detector opens development PCM/features. Acceptance remains
closed until separate acceptance protocol, reference-point selection and paired
uncertainty rules are frozen and development passes. No candidate promotion
from development alone; retain v30 and all failures.
