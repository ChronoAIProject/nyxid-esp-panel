# Reserved development execution repair, before any scores

Inherit the v1 protocol and all data/models/gates unchanged. The v1 run exited1
before inference: all90 child logs reject missing legacy frozen endpoint metadata;
zero child report.json/input.pcm artifacts. Preserve every original log.
The already pinned SLR83 manifest uses synthetic generation/RIR boundaries,
not human phonetic annotations. Add only its exact SHA256 to the lower replay
validator, matching the existing parent validator. All other manifests retain
the original metadata requirements and all PCM/path/role checks still apply.
Use short fresh /tmp/nyx-mn-slr-v2 and /tmp/nyx-mn-vox-v2 runtime paths to satisfy
AF_UNIX socket path limits. No training restart, score inspection/reselection,
audio regeneration, threshold changes or deletion/overwrite of v1 artifacts.
This is an explicit reviewed repair; errors again stop without automatic retries.
