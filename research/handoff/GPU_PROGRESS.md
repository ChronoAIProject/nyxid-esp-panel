# GPU research cross-reference

Observed 2026-11-10 at GPU branch commit
`6cf7a35525e1ee8dd28b8952307a8ed8cf61a590`:
[research-hye-nyx-voice-gpu](https://github.com/ChronoAIProject/nyxid-esp-panel/tree/research-hye-nyx-voice-gpu).
Read its subsequent commits at the beginning of each new experiment cycle.
This is an evidence snapshot, not a claim that the branch remains unchanged.

The GPU branch imports earlier Mac research from `b59c05e` and now records:

| Work | Completed evidence | Remaining evidence |
| --- | --- | --- |
| RTX 4090 / PyTorch setup | CUDA training and export on a random-array classifier | This model does not measure wake-word recognition |
| Piper CUDA synthesis | 13,920 waveform clips with full file/manifest audit | New detector fitting, streaming scores and P4 acceptance |
| Kokoro synthesis | 644 waveform clips with full file/manifest audit | New detector fitting, streaming scores and P4 acceptance |
| MicroFrontend features | 58,256 correlated sequences / 14,564 source clips; source/voice split audit passes | New model training/calibration results |
| TensorFlow GPU execution | CUDA12 library setup and real GPU matrix execution pass | Full detector metrics and P4 comparison |

The newer `6cf7a35` commit adds an audited feature pipeline, pinned Linux GPU
environment and an `official64` MicroWakeWord training command for 40,000 steps.
It has not yet published detector recognition results. This is a separate
10 ms MicroFrontend track; the current Mac continuation uses an 80 ms speech
embedding and two classifier heads. Their model inputs and trained weights are
not interchangeable. Compare actual exported detector metrics before choosing
a P4 adapter; keep each frontend/feature/calibration identity together.

The published reports contain hashes and commands; these new source WAVs and
any new trained wake weights are not attached to that branch as downloadable
assets at this snapshot. The current Mac continuation can already be reproduced
from the separate [research prerelease](https://github.com/ChronoAIProject/nyxid-esp-panel/releases/tag/research-hey-nyx-20261110)
and this branch's `assets.json`, including completed features, checkpoints and
portable plans. Its GPU environment smoke still needs execution on the target.

The two Piper runs use the same source voice ID sets: training 672, validation
112, testing 120. Whole sets were compared directly against the existing Mac
metadata. GPU resampling uses SciPy polyphase instead of the earlier
torchaudio path, so identical corpus intent does not imply identical waveform
bytes. Re-generated renderings of the same voices do not provide fresh speaker
acceptance. Preserve source voice partitions when importing any new waveforms.

The GPU branch's existing test split has already been used in earlier research
and is correctly described there as regression. The current untouched final
English/Mandarin negative corpora remain separate; no new detector scores on
those final corpora were read for this cross-reference.

Continue exchanging results by source commit, model/feature hashes, frozen
validation policy and downloadable artifacts. Keep the two active research
branches independent; compare or cherry-pick explicit changes after reviewing
their source. Production firmware and the dining-room panel remain excluded.

```sh
git fetch origin research-hye-nyx-voice-gpu
git log --oneline -8 origin/research-hye-nyx-voice-gpu
git show origin/research-hye-nyx-voice-gpu:research/wakeword_p4/GPU_RESEARCH.md
```
