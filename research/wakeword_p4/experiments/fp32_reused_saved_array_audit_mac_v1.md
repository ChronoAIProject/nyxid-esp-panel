# Independent Mac audit of the original raw-FP32 diagnostic

These opt-in readers consume only the completed, already-used development
captures. They do not run models, decode audio, select a workpoint, or read fresh
development/acceptance. They leave historical evaluators and firmware unchanged.

`intake_fp32_reused_diagnostic_mac_v1.py` authenticates every regular packet
member against the public git manifest, checks original receipts/sources/model
identities, then preserves the packet and creates an exact-byte relocated view
of the original relative capture paths. Pass a pinned publication commit:

```sh
python research/wakeword_p4/experiments/intake_fp32_reused_diagnostic_mac_v1.py \
  --repository-root REPO --gpu-commit PUBLICATION_COMMIT \
  --archive ORIGINAL_ARCHIVE --output-dir NEW_PRIVATE_DIRECTORY \
  --report NEW_INTAKE_REPORT
```

`audit_fp32_reused_saved_arrays_mac_v1.py` verifies original manifest records and
order, four authenticated model reports and JSONLs, all 1,488 original NPZ
hashes, both probability dtype/shape/domain/byte-identical repeats, full stream
lengths, every raw/clamped clock and cross-model input receipt identities. It
independently reconstructs every event and all 552 points, group denominators,
source-relative hit/missing categories and signed generated-boundary offsets.
The event oracle uses alternating high/low crossing indices over exact
binary64 `fsum` means rather than importing the GPU streaming FSM. Percentiles
use independently written linear interpolation. FAR upper95 is independently
inverted from the Poisson lower tail using `lgamma` and bisection, without
sharing SciPy's chi-square implementation. Metric comparisons allow only
`rel_tol=2e-9, abs_tol=1e-8` floating differences; events and counts must be exact.

```sh
python research/wakeword_p4/experiments/audit_fp32_reused_saved_arrays_mac_v1.py \
  --repository-root REPO --gpu-commit PUBLICATION_COMMIT \
  --capture-root NEW_PRIVATE_DIRECTORY/capture --report NEW_AUDIT_REPORT
```

The constructed self-check compares the oracle to the independently written
Mac reference, including full-window, low/high equality, cooldown/rearm,
plateau, random and short-trace cases. It also compares 200 percentile values
and checks analytic/tabulated Poisson limits plus large-count tail inversion:

```sh
python research/wakeword_p4/experiments/audit_fp32_reused_saved_arrays_mac_v1.py \
  --self-check --report NEW_CONSTRUCTED_REPORT
```

Passing constructed checks is not an actual capture audit. Passing the actual
audit proves the original saved-array diagnostic reduction, not recognition
generalization or P4 resources. Shared input/frontend hashes are authenticated
receipts, not recomputed PCM/features. Generated prefix/container boundaries
are synthesis allocations, not measured phonetic ends. All original points
and failures remain; no candidate is promoted by these readers.

## Completed original-data audit

GPU publication `2c3e7a11d7aae47e700e2f53440201c032f9584e` was consumed on
Mac. Archive `56de0b88f543a6667fab49c34115e92e022109993882c6d7155f41b6c516e5bb`
has 1,525 exact regular members, all 1,524 payload identities verified. The
independent reader passed all 1,488 traces, 552 points, 205,344 input/point
records and 20,043 emitted events. Its independent Poisson inversion differed
from original SciPy FAR bounds by at most `4.440892098500626e-16` per hour.
Original background exposure is 603,875,222 samples / 10.483944826388889 hours.
Reports are `mac-fp32-reused-diagnostic-intake-v1.json` and
`mac-fp32-reused-saved-array-independent-audit-v1.json` in `../results/`.

`describe_fp32_reused_confusions_mac_v1.py` consumes these authenticated results
and describes intersections over **all** fixed points. In Nyx float, 13 common
negative clips trigger at every one of the original 138 thresholds: nine
“Hey next”, three “Hey Nick”, one “Hey Muse”. Thus varying the threshold within
the original grid cannot remove those observed false clips. This does not
prove which training/data intervention will fix the discrimination problem.

All 360 synthetic examples use `en_GB-semaine-medium`; speed/carrier/render
variants do not create independent speakers. Willow float misses the same 28
of its 36 target clips at every registered threshold, while Willow's
integer-trained raw-FP32 model misses all 36. The official Willow comparison
also has limited coverage on this voice, per the original official report.
These observations justify source/phrase-level diagnosis before a new frozen
training intervention. They do not establish mature-model parity. The
separate pool/flat experiment and its original numerical rejections remain
separate evidence; no original gate or head is changed here.
