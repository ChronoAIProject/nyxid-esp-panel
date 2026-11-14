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

## Read the latest issue comments explicitly

The REST list endpoint defaults to an early page. Request the latest comments
explicitly when reconciling a handoff, and preserve the returned URL and body.

```sh
gh api graphql -f query='query {
  repository(owner:"ChronoAIProject",name:"nyxid-esp-panel") {
    issue(number:13) {
      body
      comments(last:10) { nodes { url createdAt author { login } body } }
    }
  }
}'
```

An individual completion comment can also be read directly:

```sh
gh api repos/ChronoAIProject/nyxid-esp-panel/issues/comments/6872754490
```

The Mac completion comments `6872754490` and `6873193380` were independently
read without an Authorization header on 2026-11-14. Both returned HTTP 200 and
the exact originally posted bodies. The authenticated latest-comment response
reports `isMinimized=false` for the direct reply; a contemporaneous GPU comment is publicly
readable. This rules out public invisibility/minimization in these checks, but
does not establish which command or cached context the other machine used.
Check current issue data before treating an older progress message as current.
Sending a result and verifying its publication do not prove the consumer read it.
