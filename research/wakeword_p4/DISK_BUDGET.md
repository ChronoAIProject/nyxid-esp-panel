# Research disk budget

Before every new download, render, feature extraction, training or export job, run `/usr/bin/python3 research/wakeword_p4/tools/check_disk_budget.py --expected-gib N`, with N covering all new outputs, temporary files, repeats, checkpoints and both compressed/uncompressed copies. The check requires 1.5 times the estimate plus a 100 GiB free reserve. A failed check stops new work until its budget is reduced or expendable files are cleaned. Check free space during long jobs as well; stop new allocations before the reserve is exhausted.

Reuse authenticated existing sources/features. Do not retain download chunks after the assembled archive is hash-verified and each chunk is verified against its byte range; keep the original archive, provenance and cleanup receipt. Estimate complete trace size before inference and use lossless compression where readers support it. Do not create redundant copies of large NPZ/WAV packs for handoff; package once and remove only verified transport duplicates.

Never automatically delete frozen v30, original native synthesis/PCM provenance, unique training inputs, original failures/interrupted jobs, final weights, or original research results. Historical execution locks remain unchanged. Cleanup receipts record exactly what was removed and why it was redundant. This policy applies to new work, including any separately declared recovery from the interrupted training run.

Completed temporary PCM replicas may share a hardlink only after SHA256 and byte-for-byte equality checks, confirming no process holds the input open. Preserve every path and all reports/logs; treat linked inputs as immutable (create a new file and replace it when regeneration is necessary). Never edit a linked input in place. Record canonical paths, hashes and reclaimed allocated bytes. Prefer one canonical immutable input for future runners.

For future experiments, store compact per-update schedule descriptors and batch digests instead of full batch tensors; retain only predeclared necessary snapshots. Review disk usage before each stage and after each long job, include scratch space on /tmp in the same budget, and check remaining space during long jobs. Do not launch new allocation stages below the 100 GiB reserve.

For new QEMU replay grids, budget scratch for the number of streams in flight. After each completed stream, verify the saved report, UART log, raw returns and stage monitor against the retained files. Once the transport header and PCM payload match the recorded settings and source WAV, the worker can release that stream's scratch directory. Preserve diagnostics from incomplete runs.

The November 22 cleanup recovered 63.37 GiB. The receipt identifies the retained model hashes and the source for each group of removed files. Its compressed index includes the archive offsets and hashes needed to recover individual ESP-IDF objects. [Cleanup receipt](results/4090-disk-cleanup-v3.json) · [File index](results/4090-disk-cleanup-v3-files.json.gz)

## Random training-feature reads

For a new Linux source-balanced trial, use the positional reader in `experiments/causal_source_balanced_pread.py` when its sampling protocol applies. It uses the same original feature files and sample schedule. Use `with Sampler(seed) as sampler` to close all handles. Frozen experiments keep their recorded implementation.

In a cold-cache test on this machine, 64 scattered spans from a private 128 MiB copy contained 2,631,680 requested bytes. Default memory mapping accounted for 134,217,728 storage bytes per pass. Positional reads with `POSIX_FADV_RANDOM` accounted for 2,899,968, with the same digest in forward and reversed method order. The private scratch file was removed. A separate check reproduced the published source-balanced ledger for the first 64 full batches (8,192 descriptors plus feature and both mask digests). Full-run speed and I/O depend on actual memory pressure. [I/O measurements](results/4090-feature-read-benchmark-v1.json) · [Sampler parity](results/4090-feature-read-sampler-parity-v1.json)
