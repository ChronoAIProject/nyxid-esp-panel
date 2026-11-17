# Current research objective — 2026-11-17

Continue synthetic-only Hye Nyx research on research-hye-nyx-voice-gpu using RTX 4090, with original C161 as the fixed development baseline. Seek a practically usable wake detector with better ordinary approximate-pronunciation, crossvoice and continuous wake-plus-command coverage, while tracking everyday word conflicts and preserving first-command audio. No human recordings or board are required for GPU research; Mac owns independent ESP32-P4 validation through Issue #18.

Keep C161 SHA256 c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81, 60952 bytes, threshold 0.91796875, smoothing 5, and the documented continuous 81600-zero-PCM priming/event protocol fixed for comparisons. Preserve models, original reports and failed results. Rare constructed nearwords are diagnostic, not a zero-error gate; do not trade normal wake hits for lower constructed-word counts. Use concrete Hey Willow references; Hi ESP is auxiliary engineering context.

Latest reviewed result: envelope candidate rejected, not promoted. Commit e1d1bb0d documents full completed coverage: no consistent benefit, Jenny targets 33/36 to 31/36, Joe 0/72 unchanged, Mac host Nyx 1/4 unchanged. Retain C161. All training/replay stages for that intervention are terminal; do not restart them.

Next milestone: review the existing untrained explicit-causal-cache prototype and its training representation, compare it against prior architecture/loss failures, and write a justified, fixed experiment protocol before starting any new training. This is a feasibility review, not a promise to train that architecture. Reject repetitive augmentation/threshold/checkpoint rescue. Then execute only a distinct justified intervention, compare with authenticated saved C161 evidence, and publish the complete result, including regressions.

Completion requires a reproducible, reviewable model and handoff supported by practical synthetic coverage and Mac/P4 numerical, command retention, memory, realtime and stability evidence. Synthetic/QEMU results alone do not establish P4 or human performance. If candidates fail, report failure and continue evidence-based work; do not mark the broad objective complete.

Publish findings promptly to the research branch and Issue #18. Use GPU for training, limit CPU concurrency (training 2, TF interop 1; deployment replay BUILTIN_REF 1 thread), reuse caches, and retain at least 100 GiB free disk. Reused background recordings are never additive independent exposure. Do not access independent acceptance material for tuning.

The existing system goal remains active. Its API permits no in-place objective replacement while unfinished; this document refreshes the execution milestones without falsely marking that goal complete.
