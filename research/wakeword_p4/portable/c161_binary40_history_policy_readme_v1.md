# Fixed C161 + Binary40 + history expert integration

This optional research candidate preserves original C161 and Binary40 and adds the exact frozen-history-teacher treatment as a third independent detector. It does not supersede the published dual-model handoff or establish P4 acceptance.

Use `c161_binary40_history_events_v1.h` together with its unchanged dependency `c161_binary40_events_v1.h`. Construct `nyx::TripleEvents` at full-stream start only. Invoke all three models on every native 480-sample/30-ms call, then feed their uint8 probability codes to `update(c161,binary40,history)`. C161 outputs uint8 directly; both explicit-cache models output signed int8 with scale1/256 and zero-128: translate `code=int(signed_output)+128`. The returned bit masks use1=C161,2=Binary40,4=history. `emitted` is zero or one bit, `detected`/`suppressed` may contain multiple bits.

All three models share the same unquantized frontend values but each model must use its own original input scale/zero-point and separately owned state. Do not feed already quantized Binary40 features to the history expert. The expert's seven cache inputs total6544bytes; initialize each to its actual real-zero code(-128 for this exact model), then copy each corresponding output after every invoke. All seven input/output quantizations match within the expert, but scales differ from original Binary40. Never share cache arrays between models. Signature tensor indices and quantizations are recorded in `4090-parallel-history-abi-original-v1.json`.

C161 internal state lifecycle, frontend and all model/decoder/arbiter state remain continuous across the81600-zeroPCM prime, clip, command and silence boundaries. Independent threshold235/256, smoothing5, refractory67calls, half-threshold rearm; shared emitted67call cooldown. Exact ties prefer C161, then Binary40, then history. Suppression does not extend shared cooldown or reset any detector. Never blend/max scores or gate whether a model is invoked on another model's probability.

The score-only fixture `4090-parallel-history-portable-scores-v1.bin.gz` has magicNYXTRI01, little-endian u32 stream count1763, then per stream u32 call count and six uint8 bytes/call: three codes in C161/Binary40/history order, then detected/emitted/suppressed masks. No clock or PCM is included. Reset only between fixture streams. Build/check from the repository root:

```sh
g++ -std=c++11 -O2 -Wall -Wextra -Werror research/wakeword_p4/portable/check_c161_binary40_history_events_v1.cpp -o /tmp/check-triple-policy
gzip -dc research/wakeword_p4/results/4090-parallel-history-portable-scores-v1.bin.gz | /tmp/check-triple-policy
```

Expected: all1763streams/3961012calls match. This tests portable event logic only, not frontend/model numerical correctness. Whole-stream uint8/nativeclock references are in the actual-runtime evidence archive on the research branch; Mac generates its own PCM numerical self-test as needed.

Three frozen files total190552bytes; distinct retained state-value payloads19108bytes; explicit cache outputs13088bytes are a separate logical payload. These counts exclude arenas, scratch, alignment, variable resources, frontend, audio buffers, queues, tasks and panel workload. They are not an RAM/arena estimate. On P4 first check resolver/version support and exact outputs, then all three resident arenas and whole-panel internal/PSRAM peaks, per-call scheduling against30ms, first-command preservation and sustained operation. Preserve unknowns/load failures instead of declaring resources feasible from file size. C161 needs resource-variable initialization operators; both other models have ordinary explicit caches and seven listed INT8 op kinds, all recorded in the ABI reports.

Recognition remains incomplete: continuous new builtin/US/BR11/12,11/12,9/12; Joe23/72, MeloBR25/36, Mac host2/4; all are consumed synthetic diagnostics. Extra Muse/Mix/Nick non-target clips, daily Nick1/96 and reused Melo background1event remain disclosed. Rare constructed nearwords are diagnostic, while common-name Nick is a practical conflict. No new independent voice/background exposure or human generalization claim. Use Issue#18 for separate P4 numerical/resources/realtime/first-command/stability evidence with firmware commit, model hashes and fixed profile.
