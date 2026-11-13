# Official Hey references: artifact identity and P4 build preparation

This is preparation for the new Issue #13 comparison, not a recognition result.
The three primary arms are official Hey Willow, our method trained on the same
Hey Willow phrase, and our method trained on Hey Nyx with a matched recipe.
Hi/Hey Muse is an optional closer word-length reference; Hi ESP remains the
system reference. A same-phrase comparison removes the changed phrase but does
not equate the proprietary and custom architectures, training sets or frontends.
The two custom arms help assess phrase difficulty within our method.

Before scoring, freeze sources/roles, paired text and environment generation,
confuser/cross-phrase labels, calibration and event arithmetic, and the independent
acceptance boundary in the Issue. Compare at a shared false-wake target, not a
shared numeric threshold. Preserve all original failures and current formal runs.
No speech has been generated or scored for this new protocol by these checks.

## Immutable artifact inventory

ESP-SR is already pinned to 2.6.1, registry hash
`7ba1e97d446440faa96a7d98fbda7dbaaf7ae6b960e06cc68264d38aa6eb4d14`,
original commit `b9dc43a329a68de2f72147808fc9ea29d2ace71e`.
Both official Hey models are present; no SDK upgrade is needed to obtain them.

`inventory_official_hey_models_v1.py` independently compares 19 listed installed
files with immutable upstream bytes. The registry rewrites the component manifest:
its normalized YAML fields match after excluding registry repository annotations;
that manifest is not byte-identical. All other listed files match byte for byte.
This does not authenticate every file in the SDK. The P4 library/header scope is
revision <3 with IDF5.

| Official directory | Raw members | Packed bytes | Layout |
|---|---:|---:|---|
| `wn9_hiesp` | 290982 | 291142 | metadata, index, data |
| `wn9_heywillow_tts` | 290880 | 291040 | metadata, index, data |
| `wn10_himuse` | 817508 | 817708 | two metadata/data pairs |

Hey Willow metadata reads `wakenet9l_tts1h8_Hey,Willow_3_0.644_0.650`.
Retain the official `wn9_heywillow_tts` name and public factory dispatch; the
metadata label alone does not establish the actual interface, topology or runtime
cost. Muse includes both Hi/Hey prefixes in each metadata pair. Suffix numbers
are not measured threshold getters or calibrated comparison points.

The inventory runs the original upstream packer on isolated copies and unpacks
every member to verify names, offsets, coverage and exact contents. The upstream
packer enumerates filesystem order: its packed SHA identifies this concrete
output, not one canonical encoding. Our existing OTA embedding tool sorts member
names. Its expanded header order/hash differs while the model name and every
member byte agree. The initial host check incorrectly required identical full
packed hashes and failed; the corrected check verifies both complete layouts
and exact named contents, without changing either packer's ordering or weights.

## Opt-in probe and verification

The standalone [baseline](../../wakeword_baseline/README.md) adds explicit
`BASELINE_WAKE_MODEL=wn9_heywillow_tts` selection. The original Hi ESP option
retains direct WakeNet9 interface selection and recreate/zero-prime/full-PCM/tail
lifecycle. The model embedding tool now requires its selected model name to match
the source directory, so a Willow directory cannot be silently packed as Hi ESP.
Its default Hi ESP output is byte-identical to `67e4896`.

The guarded runner verifies actual ready-message engine/mode/rate before sending
PCM and after baud negotiation. The historical Hi ESP summarizer rejects Willow
reports rather than applying Hi ESP labels. Host checks cover both real official
model layouts, legacy default byte identity and nine explicit rejection cases.

Use an environment with PyYAML and pyserial (the pinned IDF environment here has
both). Output directories must be fresh and outside git:

```sh
python research/wakeword_p4/experiments/inventory_official_hey_models_v1.py \
  --output "$PRIVATE_INVENTORY"
python research/wakeword_p4/experiments/check_official_hey_reference_v1.py \
  --inventory "$PRIVATE_INVENTORY" --output "$PRIVATE_HOST_CHECK"
```

For P4 compilation, use pinned ESP-IDF 5.5.4 and separate build/sdkconfig paths as
documented in the baseline README. The recorded build copies the exact project,
lockfile, embedding tool and six locked managed components into a private tree
before compiling each selection. It does not reconfigure the full panel app.
The build report records source/artifact identities and verifies the embedded
compressed model against the host-checked bytes. It records compilation/linking
only. Neither compiler success nor a loaded-model ready message establishes a
successful create/detect call, sample cadence, memory use, recognition or latency.

Both builds completed with the pinned RISC-V 14.2.0 toolchain. The Hi ESP app is
680336 bytes; the Willow app is 1001072 bytes. Both compiler warning counts are
zero. Willow has one partition-size warning: the standalone project's 1 MiB
factory partition has only about 5% free. This is not a measurement of the
provisioned board's active OTA slot; the guarded runner must still verify that
slot independently. The larger Willow app includes factory dispatch and a
different compressed model; the binary-size difference is not model runtime RAM.

To verify the recorded private build tree (layout in the recorder's docstring):

```sh
python research/wakeword_p4/experiments/record_official_hey_build_v1.py \
  --build-root "$PRIVATE_BUILD_ROOT" --host-check "$PRIVATE_HOST_CHECK" \
  --output "$PRIVATE_BUILD_REPORT"
```

Muse is inventoried but deliberately unsupported by this probe: its two-pair
layout and 817708-byte expansion need a distinct reviewed runtime path. Willow
fits the unchanged 512 KiB expansion bound. Do not infer Muse compatibility from
a Willow build.

Reports live in `research/wakeword_p4/results/official-hey-*-v1.json`.
No model binaries, SDK implementation/disassembly, speech, board addresses,
credentials or private logs are included. The original inventory/build stage
operated no board, speaker or microphone, opened no final/acceptance set and
promoted no candidate. The subsequent [native zero-PCM check](official_hey_zero_smoke_v1.md)
records actual P4 initialization/resources and its separate limits/restoration.

## Separate paired-v2 crop review

GPU commit `ece33790ac48b42a285b8cccb0cab4f0afdda5c2` retains the v1
pre-training prefix failure and implements the separately registered v2
geometry cap without modifying original audio. Mac independently extracted
that exact commit's two CPU analytic tests and seven source files, and ran
`python3 research/wakeword_p4/experiments/test_hey_prefix_crop_v2.py -v`.
Both tests passed on NumPy 2.4.3. Ten actual sampler batches on analytic
features preserve labels, weights, every negative tensor and all three RNG
states; 320 original/effective positive draws are recorded, with changes
restricted to the registered cap. The receipt
`results/mac-hey-prefix-v2-analytic-check-v1.json` binds all seven source hashes.
This is not actual new-pack CPU/full-schedule/GPU/recognition admission or
cross-platform numerical equivalence. Reproduce from that GPU commit in an
isolated checkout; no TensorFlow, waveform, board or reserved data is needed.

## Reserved-development contract review

Mac independently compared GPU commit `ec127683f508e0352267b16dd876992ecef7c3ba`
render contract against the original generation-plan-v2 reservations. All120
original development request objects and the complete voice metadata match,
including texts/phones/token IDs/scales/seeds. The contract retains360rendered
cells/36positives per target, and its noise-seed set is disjoint from training.
The result `results/mac-hey-development-contract-review-v1.json` binds both
source JSON hashes. This is a metadata-only check, without new waveforms,
detector scoring, runtime or acceptance admission. Fresh RIR registration and
mechanical quality alone do not establish recognition or source independence.
