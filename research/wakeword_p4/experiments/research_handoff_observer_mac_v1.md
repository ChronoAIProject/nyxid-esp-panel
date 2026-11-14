# Read-only handoff observation

Run one observer while the original GPU tasks continue. Use a previously saved
`gh release view ... --json assets` response as the baseline. The process checks
the existing GPU branch for complete one-way formal/compile results and the
existing release for newly added old-regression/event assets. It writes an
append-only observation journal and exits for handoff inspection when either
becomes available; asset names are discovery hints, not authentication.

```sh
python3 research/wakeword_p4/experiments/observe_research_handoff_mac_v1.py \
  --repository-root . \
  --baseline-assets EXISTING_RELEASE_ASSETS.json \
  --output-dir NEW_PRIVATE_DIRECTORY \
  --interval 45 --max-observations 80
```

Poll the same returned tool session while it is live. This proves only the local
read-only observer is running; every journal entry explicitly denies verifying
GPU process liveness. Read errors and observation-window expiry do not imply GPU
termination and must never cause training or a collector to restart. The earlier
bounded observer `49446` is terminal and is not reused or restarted.

No messages, payload downloads, inference, scoring, private-array reads or model
qualification occur here. When an event asset appears, authenticate its immutable
manifest and original report hash chain before feeding its original 1243 payloads
to the frozen `9162bdb` reader. When models appear, inspect their actual identities
and complete numerical gates before opening any new development or acceptance.
