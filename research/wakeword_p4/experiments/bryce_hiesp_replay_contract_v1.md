Bryce synthetic-duration Hi ESP replay contract v1
================================================

This handoff prepares official replay. It contains no Hi ESP scores, engine
binaries or actual GPU execution profile. No QEMU, audio playback, microphone,
board/firmware or model inference was used for its Mac validation.

The126original WAV/PCM hashes are unchanged. The derived manifest is
`a14837a2564e6f5bd10aee89e991568bc05827770b750498843a23eecd47b2c2`.
It retains all sources in order and declares36HeyNyx /36HiESP /54common-confuser
rows. Original confuser identities remain in `source_target`; only the paired
classification `target` is null for Nick/Next/Mix. This is consumed synthetic
development, never training, final or certified independent human speech.

`keyword_end` is the generator token-duration prefix end before punctuation,
not command end or a human phonetic endpoint. The source manifest, generation
plan and independent original reconstruction audit are byte-bound. The helper
rejects missing/reordered/relabeled sources and changed endpoint semantics.
Do not invent `voice_partition_sha256` or `endpoint_annotation_source_sha256`.

The15-added/1-replaced-line patch targets the GPU runner from8e2d1cb/a212113:

- original runner SHA `9f4e561c4ee78c52099610084e418fe524cab902d124943898c6bb76f970c1e6`;
- patched runner SHA `049643a2d92d8a7cc2b2a13ef3102bdf1bc5fa3fd9517ec31be20ba98ce1b3fe`;
- patch SHA `e813317940009cbfb7ddf7c87e5f02d81869e6725220ec7a887916e73308e018`.

The new schema requires its own `--development-preregistration`; the helper
loads only when that flag is supplied. The preregistration binds the complete
manifest, source provenance, helper/runner bytes, actualQEMU/ELF/flash hashes,
historical.63 requested/.62496001getter policy,512chunk,DET_MODE_95 and no
extra prime. The original final gates and audio/QEMU/CRC code blocks remain
byte-identical. No firmware, detector arithmetic or model changes are made.

The first draft failed the negative test “omit the new flag and insert legacy
fields”: it reached the intercepted launch boundary. That failure is retained;
the final patch explicitly requires a preregistration for the new schema.
All24final contract cases pass. Invalid inputs reject before nm/QEMU/output
creation; valid admission is tested by intercepting the launch boundary.
An unrelated integer-ramp unit WAV checks old transport admission and exact
transferred PCM. No real cohort was scored through the transport role.
The engine/reference fixtures are non-executable test bytes, not real engines;
these tests never prove official inference or actual GPU binary identity.

Receive and prepare in a separate evaluation directory
----------------------------------------------------

The package includes a standalone source tree with the exact patched runner;
using it does not change any pinned source in the current formal training.
The patch is also provided for review; do not apply it to an executing locked
worktree. Reuse the existing official binary files unchanged.

Download/extract `heldout-bryce-development-v1.tar.gz` from the same research
release, then materialize the replay directory:

```sh
python source/research/wakeword_p4/experiments/prepare_bryce_hiesp_replay_v1.py \
  --data /path/to/heldout-bryce/data \
  --audit /path/to/heldout-bryce/independent-audit-v1.json \
  --output /fresh/local/bryce-replay
```

Require the manifest SHA above. The file `reference/manifest.json` in this
handoff is an exact expected metadata snapshot, with no duplicated WAVs.
The preparation command creates the actual126byte-exact WAV copies.

Repeat the code/admission audit without running any actual engine:

```sh
python source/research/wakeword_p4/experiments/audit_bryce_hiesp_replay_contract_v1.py \
  --inputs /fresh/local/bryce-replay \
  --base-runner reference/base-run_replay.py \
  --patched-runner source/research/wakeword_p4/qemu_esp32_replay/run_replay.py \
  --output /fresh/local/contract-audit
```

Freeze actual binary/profile bytes before official inference
----------------------------------------------------------

Use an existing completed official report (or original report list), not the
unit-fixture reference. The freeze command checks source hashes and that the
actual binary files match every supplied completed reference. It only writes
the profile and does not launch a child detector. Preserve failed artifacts.

```sh
python source/research/wakeword_p4/experiments/freeze_bryce_hiesp_replay_v1.py \
  --manifest /fresh/local/bryce-replay/manifest.json \
  --runner source/research/wakeword_p4/qemu_esp32_replay/run_replay.py \
  --qemu /existing/qemu-system-xtensa \
  --elf /existing/nyx_wakenet_esp32_replay.elf \
  --flash /existing/qemu-flash.bin \
  --engine-reference /existing/4090-phone-development-official-replay-reports-v1.json \
  --output /fresh/local/bryce-hiesp-frozen-profile.json
```

Commit/push and report this actual profile/hash in Issue13 before first
inference. Previous reported binary identities are QEMU`c15716f0…`,
ELF`cc511f55…`,flash`7e140bd5…`; the command must check actual files.
Do not label the Mac fixture profile as actual GPU/engine verification.

Then invoke the patched runner for **every** original manifest row, using
the same exact frozen inputs. For example, one child invocation is:

```sh
python source/research/wakeword_p4/qemu_esp32_replay/run_replay.py \
  --manifest /fresh/local/bryce-replay/manifest.json \
  --development-preregistration /fresh/local/bryce-hiesp-frozen-profile.json \
  --record wav/kristin-0-hey_nyx-quiet.wav --threshold .63 \
  --qemu /existing/qemu-system-xtensa \
  --elf /existing/nyx_wakenet_esp32_replay.elf \
  --flash /existing/qemu-flash.bin --nm /existing/xtensa-esp32-elf-nm \
  --output /tmp/nyx-bryce-esp-v1/0
```

Use a short fresh runtime path for AF_UNIX. Existing four-worker child
scheduling can be reused; retain all126logs/reports in original manifest
order. A failed/missing child invalidates the complete summary, never select
successful clips. Require every stage104,CRC,sourcePCM hash,threshold getter
and fixed engine/runner/helper/profile identity. No automatic retry/restart.

Do not pass paired positives through `run_corpus.py`'s negative/FAR summary.
Do not replace the retained v30 application-prime baseline with a newly
unprimed Nyx run from `compare_paired_validation.py`. Send the raw126official
reports; they can be joined to the existing frozen Mac v30 result by original
source key/condition and exactPCM hash without more Nyx inference.

Report HiESP whole-source any-event and prefix[-100,+800]ms windows separately,
alongside voice/condition/bare-command groups, all common/confuser/opposite
events and signed reference timing. Thresholds are independently fixed and no
background is included in this paired cohort, so no matched-FAR or independent
acceptance claim. The separate12.234h Earnings22 development is a later
samePCM background stage. All existing final sets and production boards stay
excluded; passing admission/unit tests never establishes HiESP parity.
