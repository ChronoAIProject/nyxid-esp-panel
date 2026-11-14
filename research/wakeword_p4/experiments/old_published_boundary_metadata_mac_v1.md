# Published old-regression metadata reduction

This descriptive reduction uses only the immutable original metrics and GPU
four-boundary supplement at `ca79386`. It authenticates both exact published
JSON hashes, retains all 1242 existing fixed points, checks the event identity
and 36 positive rows per point, and reproduces every original dry-prefix count.
It does not read original event NPZs, scores, audio, weights or new material.

```sh
python3 research/wakeword_p4/experiments/reduce_old_published_boundary_metadata_mac_v1.py \
  --repository-root . \
  --report NEW_REPORT.json
```

Across all 138 fixed points, the old Willow float-control float path has any
event in only 7–8 of 36 positives, and UINT8 in exactly 7; at most two clips
have events only before the generated dry-prefix boundary. The old Willow
integer-QAT float and UINT8 paths have no event in all 36 positives at every
point. Thus the published metadata does not support explaining most old Willow
misses solely by a timing-boundary choice or solely by UINT8 conversion. It
does not establish a training root cause or measure current bounded-family
recognition. All no-event clips and all original failures remain in the report.

These ranges describe historical consumed data and select no operating point.
The four generated source references are not human phonetic boundaries. The
separate frozen `9162bdb` reader must still authenticate and read all 1242 actual
event NPZs plus the original source manifest. This report cannot replace that
work, numerical admission, matched official comparison or independent acceptance.

Mac prepared a detached worktree at `9162bdb` and materialized the exact frozen
parent plan and original collection/metrics/audit. Their hashes match the public
completion chain. Original 1243 private payloads (manifest plus event NPZs) were
requested through Issue #13; no model collector or reducer is restarted.
