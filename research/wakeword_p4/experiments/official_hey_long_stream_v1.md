# Explicit native complete streams up to two hours

The paired study's already frozen development background includes three whole
sources longer than the old one-hour cap: 68499840, 58641024 and 61266240 samples.
The longest is 4281.24 seconds. Reserved acceptance metadata, without opening
media or transcripts, reports at most 6269 seconds. Complete-source tests must
preserve detector/frontend state throughout a recording, so splitting/resetting
these sources would change the experiment.

`BASELINE_LONG_STREAMS` defaults OFF and requires threshold control ON. Its
separate `official-long-stream-v1` ready handshake declares an inclusive
115200000-sample / 7200-second limit. The host requires explicit `--long-streams
--threshold-control`; it rejects mismatched firmware or limits before sending
PCM. Without this option the original legacy/threshold command formats and
one-hour controlled limit remain. Model bytes, factory/mode, per-stream create,
63 zero-prime calls, raw event accounting and final-only padding are unchanged.
The full-event buffer still allocates one entry per possible input call, at most
225000 entries / 1800000 bytes in PSRAM. No input processing is reset midstream.
Periodic host progress counts bytes submitted to serial, not detector-consumed
samples; only final CRC/sample/call verification authenticates consumption.

The preparer freezes two streams before inference:

1. 16001 exact-zero samples, exercising a final partial chunk.
2. Exactly 115200000 samples. The original consumed Samantha bare-Willow
   container occurs verbatim at sample 0 and at sample 115144491. Every sample
   between the two complete containers is exactly zero. The source's own silence
   is retained. This is a transport/state/clock fixture, not a natural recording
   or two-hour negative exposure for FAR.

Use the already published twelve-file source manifest
`cb536c6e9c2176c51a88d3af6a1867ad00602053b0b4d1f788294acf8cbd8b2e`.
No new synthesis, resampling, gain change, trim, playback, microphone or reserved
material is involved. Only official Willow DEFAULT/no setter is requested;
there is no threshold sweep or selection. Transfer baud is fixed to 1500000.
All prime returns, input return counts, complete positive-event clocks, build
identity and restoration fields are retained. Capacity and a positive event
after the original one-hour boundary are reported separately. Preserve any
failure/missing late event; never extend, regenerate or tune this fixture.

Host checks compare command bytes/handshakes with `00b36d4`, test both engines,
exact old/new boundaries, invalid counts/mismatched opt-in and constructed
two-hour event clocks. Default OFF source preprocessing/compilation and actual
linked products are recorded separately. These are implementation checks, not
native two-hour completion until the full guarded run ends successfully.

```sh
python research/wakeword_p4/experiments/official_hey_long_stream_v1.py check-host \
  --output "$PRIVATE_LONG_HOST"
python research/wakeword_p4/experiments/official_hey_long_stream_v1.py prepare \
  --source-fixtures "$CONSUMED_SPEECH_FIXTURES" \
  --build-report "$PRIVATE_LONG_BUILD_REPORT" --output "$PRIVATE_LONG_FIXTURES"
```

Commit the helper/protocol before running the new probe on the freshly verified
USB development panel. Use the guarded baseline runner's complete original
MAC/app/partition/slot/restore arguments, plus `--engine wn9_heywillow_tts
--require-resources --threshold-control --long-streams --baud 1500000`.
The existing offline development configuration uses explicit `--restore-readiness
app-only`; original app/slot/service/idle/Hi ESP listener and independent final
HTTP status must be verified after restoration. Do not use standalone `idf.py
flash`; never operate the dining-room production panel. A timeout/poll gap does
not authorize restarting an active run.

```sh
python research/wakeword_p4/experiments/official_hey_long_stream_v1.py audit \
  --fixtures "$PRIVATE_LONG_FIXTURES" --report "$PRIVATE_LONG_RUN/report.json" \
  --build-report "$PRIVATE_LONG_BUILD_REPORT" \
  --expected-restored-elf "$ORIGINAL_ELF_SHA" --output "$PRIVATE_LONG_AUDIT"
```

The auditor checks both complete WAVs and the verbatim marker/zero construction,
all authenticated source/probe/request/getter/CRC/sample/call/event and restoration
fields. A capacity result does not admit the fresh paired study's material or
prove recognition, FAR, phonetic latency, independent parity or voice connectivity.
Original frozen speech-grid and historical evidence stay unchanged; their public
archives retain the exact original sources for reproduction.

## Pre-inference frozen implementation checks

`results/official-hey-long-stream-plan-v1.json` and the adjacent manifest bind the
exact two fixtures and source/build identities before native inference. The
manifest SHA is `d7f1ec9a0271ad34b8fb5ad61470b6c2e3535cca0b27cff8881046903307ac6a`;
protocol SHA is `68d3751cb8ca47cc1575ab7d6bb7ab50db791c31b2c26dc125620b5b44ab398b`.
The generated 230.4 MB PCM is reproducible with `prepare`; it is not committed.

Adjacent `host-check`, `compile-check` and `build` reports record completed
pre-inference checks. Eight OFF-profile preprocessing comparisons differ only
in four ESP_ERROR_CHECK diagnostic line constants; the eight current OFF
profiles and four ON profiles compile without compiler warnings. Both C invalid
profiles and all four actual CMake guard combinations behave as specified.
This is not a claim that complete app binaries are byte-identical. Both isolated
ON-profile P4 builds succeeded with the original embedded model bytes. Willow
has the same standalone 1 MiB partition space warning; the guarded native run
uses the separately verified, much larger active application slot.

Native execution and app restoration have not yet been claimed by this frozen
plan. They require the complete report, saved-frame audit and independent
post-restoration checks.

## Observed host-drain failure and separate identical-input retry

The original frozen `8841d6f` run completed the short control but blocked after
all long PCM writes in macOS WCH `termios_tcdrain_impl -> tcdrain -> ioctl`.
A one-second native stack sample confirmed that blocking site; an independent
queue query also blocked. No long RESULT/CRC/event was observed. After this
concrete host diagnosis, the query and original task received SIGTERM so the
original guarded task entered restoration and exited130. This was not a restart
on an observation timeout. The original report, five detector frames, host
failure receipt and independent restore check are retained in the adjacent
`official-hey-long-stream-failed-*` / `host-drain-failure` / `failure-restore`
results. Original app/slot/service/agent/HiESP listener and no serial owner were
independently confirmed. Nyx remained connecting. No native capacity or late
recognition pass is inferred from submitted bytes or the failed report.

The separate retry keeps the exact same WAVs/manifest, model, DEFAULT threshold,
firmware and UART baud. Only explicit long-mode host completion changes to
`board-result-ack-no-tcdrain-v1`: after successful ordered writes, read the
board's existing complete result directly instead of invoking OS tcdrain.
The final sample/call/CRC/event checks remain mandatory and authenticate
consumption independently of host queue drain. Legacy mode retains its original
flush. The new runner also explicitly prints pcm_submitted, which still means
submitted bytes, not consumed samples. This is a transport correction, not a
new source, threshold rescue or selection. The retry plan and exact new source
hashes must be pushed before executing a fresh guarded run.

The separately frozen retry is `results/official-hey-long-stream-retry-plan-v1.json`,
SHA `09536af0def85b666b445c1f50b3578e0f9bfb95fa36ded51d6194fba1a912b2`.
The complete WAV files and manifest compare byte-identically with the failed
first run; model/build/C source identities are unchanged. Adjacent retry host
checks preserve original command/handshake and boundary behavior. No C rebuild
is needed for this host-only correction. Real duplex completion still requires
the new guarded run and full independent result/frame/restore checks.

## Completed native identical-input retry

The original retry session49535 terminated normally with exit0. Both complete
inputs passed sample/call/CRC/getter/full-event accounting. The long stream
consumed all115200000 samples in225000 input calls, CRC32 `643a9139`, without
intermediate reset. Its two positive events ended at samples39936 and115184640;
the second is after the previous one-hour limit. The16001-zero control used32
calls and no positive events. Both streams'126 prime returns and all8 saved
detector frames were independently audited against the unchanged full WAVs.
A separate constructed copy omitting one real positive event was rejected by
the independent auditor; original report/frames remain unchanged.

Original app/ota_0, selected service and agent, voice idle, HiESP listening and
absence of a serial owner were independently verified after the guarded restore.
Nyx remains connecting as before. This is app-only restoration and complete
continuous-stream capacity/late-event evidence, not voice connectivity, two-hour
negative FAR exposure, new-source recognition or parity acceptance. No new
candidate or threshold was selected; the prior host-drain failure remains.

The public `official-hey-long-stream-native-v1.tar.gz` research asset is342816B,
SHA256 `be495ac644f3e8b215a9970bf90a0f0896545ef6ac11db667a84ab21ec0006de`.
It includes both full WAVs, marker, exact retry/failure protocols and sources,
original reports, independent saved frames/audits/restoration and reproduction
commands; no firmware/models/private logs/network settings/credentials. Anonymous
redownload verified all33 payload sizes/hashes; both included auditors reproduced
their original summaries byte-identically. Adjacent `official-hey-long-stream-`
native/frame/restore/archive/public-download JSON files retain these results.
