# Original aligned-prefix candidate for Mac/P4 functionality evaluation

Download:
https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/gpu-causal-aligned-prefix-mac-v1.tar.gz

Archive:35078bytes, SHA256
`ccda3308b3891163a1a6bb369ea131d70fc26db7d408b283f4c4bef72f25b319`.

Only two members: original `hey_nyx.tflite` and `manifest.json`.
The model is60952bytes, SHA256
`3ea7b067f1000516c8dfe9c6005bccdbe7360a2c26fc3d7790b70e5d26e8d3b7`.
No re-export, audio, credentials or numeric selftest is included.

Simple profile: context309, threshold0.91796875, smoothing_steps5, stride3.
InputINT8[1,3,40]; outputUINT8[1,1], scale1/256, zero_point0.
Use the existing frontend, event policy and81600sample continuous priming.
Mac has the experimental streaming profile integration; no teacher60 report,
calibration admission or exact-score requirement is needed for this handoff.

The archive was downloaded anonymously without an authentication header.
Archive size/hash and every member size/hash match; model bytes equal the
original single export. Receipts and exact manifest are in
`../results/4090-causal-aligned-prefix-mac-anonymous-download-v1.json` and
`../results/4090-causal-aligned-prefix-mac-manifest-v1.json`.

Purpose: actual isolated development P4 operation, connection and stability
feedback. This experimental handoff does not replace frozenv30 or establish
independent acceptance. Known consumed results remain: commands24/27,
light6/9 with generated-reference hit median~133ms, common17/288, background
0/10.483944826h. Generated-end early bare events alone are not failures and
are not phonetic truth. No new perfect-model or numerical agreement gate.
