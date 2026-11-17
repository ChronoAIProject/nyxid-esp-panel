# Current research objective — 2026-11-17

Continue synthetic-only Hye Nyx research on research-hye-nyx-voice-gpu using RTX 4090, with original C161 as the fixed development baseline. Seek a practically usable wake detector with better ordinary approximate-pronunciation, crossvoice and continuous wake-plus-command coverage, while tracking everyday word conflicts and preserving first-command audio. No human recordings or board are required for GPU research; Mac owns independent ESP32-P4 validation through Issue #18.

Keep C161 SHA256 c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81, 60952 bytes, threshold 0.91796875, smoothing 5, and the documented continuous 81600-zero-PCM priming/event protocol fixed for comparisons. Preserve models, original reports and failed results. Rare constructed nearwords are diagnostic, not a zero-error gate; do not trade normal wake hits for lower constructed-word counts. Use concrete Hey Willow references; Hi ESP is auxiliary engineering context.

Latest reviewed result: Binary40 final trained/exported/complete consumed replay is preserved in binary40_findings_v1.md (631ff6d1). It improves builtin5→8/12, continuous0→7/12 and2→8/12, Legacy35→36/36 and Joe0→17/72, while trading normal Jenny hits, losing one bare hit per Melo voice, adding one daily Nick-address event and one Melo background event. Mac host1→2/4 is not P4. Retain C161 baseline and Binary40 research candidate, no automatic promotion. Envelope failure and prior cache/unit training history remain preserved; do not restart completed runs.

The background context audit supports an ordinary-speech concern, with ASR limitations retained. Original full128 training-batch fit audit finds32/32 positive crossings and0/96 negative events for both float/INT8. Largest probability difference0.854 occurs in an unsupervised interval; five event timings differ. These narrow training results do not establish generalization or full-stream quantization equivalence. See binary40_training_fit_findings_v1.md. Next evidence should establish whether normal consumed crossvoice regressions are already present in the final float model, with complete fixed scope and original physical priming, before selecting any distinct intervention. No new training, threshold/calibration rescue or repeat of historical QAT/unit/augmentation recipes is selected.

Completion requires a reproducible, reviewable model and handoff supported by practical synthetic coverage and Mac/P4 numerical, command retention, memory, realtime and stability evidence. Synthetic/QEMU results alone do not establish P4 or human performance. If candidates fail, report failure and continue evidence-based work; do not mark the broad objective complete.

Publish findings promptly to the research branch and Issue #18. Use GPU for training, limit CPU concurrency (training 2, TF interop 1; deployment replay BUILTIN_REF 1 thread), reuse caches, and retain at least 100 GiB free disk. Reused background recordings are never additive independent exposure. Do not access independent acceptance material for tuning.

The existing system goal remains active. Its API permits no in-place objective replacement while unfinished; this document refreshes the execution milestones without falsely marking that goal complete.

## Refreshed execution milestones — 2026-11-17

User requested a goal refresh. The broad system goal remains active; its API cannot replace an unfinished objective. Continue toward the same practical outcome with these immediate milestones:

1. Finish the final-weight float replay evidence audit. The v1 replay omitted checkpoint loading and is invalid; retain its invalidation record. The v2 final-weight metrics and native/Keras check passed, but duplicate trace filenames overwrote some files. Count affected traces from original hashes, recover only affected traces with unique phase/index names, and authenticate the complete evidence archive before claiming completion.
2. Publish the reviewed float/INT8 comparison, corrections and original artifacts to this branch and Issue #18. Preliminary valid metrics show several everyday errors and normal pronunciation misses already exist in float; do not treat quantization as their sole cause or promote Binary40 automatically.
3. Select the next distinct intervention from the verified evidence and historical experiments, targeting ordinary approximate wake pronunciation, crossvoice connected commands and everyday phrase conflicts. Use RTX 4090 for training; keep C161 and the fixed comparison protocol. No new human recordings, board dependency, threshold sweep or repeated failed recipe.
4. Preserve the original model/report bytes and communicate with Mac through Issue #18 for P4 numerical agreement, first-command retention, resources, realtime and stability validation. Push completed research promptly; retain at least 100 GiB free and the existing CPU concurrency limits.

Success remains a reproducible, useful candidate supported by practical synthetic evidence and the separate Mac/P4 checks, rather than a requirement to reject every unusual nearword. No candidate is declared accepted by this goal refresh.
