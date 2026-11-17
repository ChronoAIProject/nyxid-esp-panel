# C161 + original Binary40 event policy

This portable C++11 header implements the fixed two-independent-detector event policy. Model/frontends are not included in this policy selftest. C161 outputs uint8 directly; convert Binary40 signed int8 probability with signed+128 before passing it to update. Both use the exact1/256 probability lattice.

Feed one pair per native480sample call (30ms at16kHz), including all81600zeroPCM prime calls and original stream finish calls. Maintain both inference states and the ParallelEvents object across prime/audio/clip boundaries. Construct a new policy only at full stream start. Each detector has5code smoothing, threshold235/256=.91796875, half-threshold rearm and67call own refractory. Shared output cooldown67calls, C161 priority for simultaneous events; suppression extends no cooldown and resets neither detector. detected/emitted/suppressed usebit0=C161,bit1=Binary40.

On Mac or Linux from repository root:

```sh
c++ -std=c++11 -O2 -Wall -Wextra -Werror research/wakeword_p4/portable/check_c161_binary40_events_v1.cpp -o /tmp/nyx-policy-check
gzip -dc research/wakeword_p4/results/4090-c161-binary40-portable-policy-scores-v1.bin.gz | /tmp/nyx-policy-check
```

Expected output: All1760streams,3941335calls match saved events exactly (with spaces as printed by the checker). Both compilation and checker must exit0. Fixture SHA256709c6f7eb8c6a31af84f38cb63ba4a74d6ecbbfc37842f0a8b8ab1f27d82fc97,77590bytes. Fixture contains raw probability codes and expected policy bits, **no PCM**. Binary format: ASCII NYXPAR01, little-endian u32 stream count; each stream has little-endian u32 call count, then fiveuint8percall: C161code, B40code, detectorbits, emittedbits, suppressedbits. Complete1760saved consumed streams are retained with per-stream policy resets, no waveform truncation.

This check proves portable event arithmetic against saved score references. It does not exercise either model or establish P4 numerical inference, arena/whole-panel memory, realtime, first-command retention, stability or usable independent recognition. Models remain experimental as a parallel system; C161 is still baseline. See c161_binary40_parallel_findings_v1.md for current gains/conflicts and actual-host-runtime verification status.
