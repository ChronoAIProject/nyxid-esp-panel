# Independent saved-window and prospective-decision audits

Mac independently checked the authenticated GPU first-10,000-batch trace for
each original word against the original training metadata. The tools do not
import the original sampler, read feature arrays, invoke models or access media.
GPU's independent coordinate-to-tensor reconstruction remains separate evidence;
the Mac audit authenticates its report and source, without claiming to repeat it.

Inputs are the published original metadata package and surviving-window package:

- `gpu-original-broad-training-metadata-mac-v1.tar.gz`, SHA256
  `fe15ab839bb6df3f17b264f6b84bd48f5d6199f7665bbac4aec75ddef24d0e9b`.
- `gpu-original-broad-surviving-window-census-v2.tar.gz`, SHA256
  `1b5f0d1079797c0f80fc245b559259c377a70520d29a4d952bf27c51144b26bb`.

Both are assets of the `research-hey-nyx-20261110` release. Authenticate the
archives before extraction; the first tool additionally checks the pinned
trace manifest, exact 13-file set and every payload, original git source
identities, and metadata/provenance hashes. Use new report paths:

```sh
python3 research/wakeword_p4/experiments/audit_hey_broad_saved_windows_mac_v1.py \
  --repository-root REPO --packet TRACE_PACKET \
  --original-metadata METADATA_PACKET --report NEW_COORDINATE_REPORT.json

PYTHON_WITH_NUMPY_2_4_6 research/wakeword_p4/experiments/audit_common_carrier_decisions_mac_v1.py \
  --repository-root REPO --packet TRACE_PACKET \
  --original-metadata METADATA_PACKET --coordinate-audit NEW_COORDINATE_REPORT.json \
  --report NEW_DECISION_REPORT.json
```

## Executed results

The original coordinate-audit session `73153` and decision-audit session `54306`
both exited 0. Each word has 1,280,000 surviving windows: 320,000 positives,
320,000 ordinary negatives, 160,000 own-prefix negatives and 480,000 backgrounds.
An additional 160,000 overwritten ordinary draws per word were validated but
excluded from surviving coverage. Every grouped count, source/conditioning
cardinality and generated-boundary count matches the GPU result exactly.

All 227,714 surviving common negatives per word are bare. The reciprocal
negatives include 49,311 command-carrier windows and 42,975 bare windows.
Positive carriers are bare160,051/light53,378/time53,272/timer53,299. These are
effective window counts, not independent speakers or proof of pronunciation.

At matching shuffled positions, both words have identical source-pair
conditioning/carrier/variant identities. All 480,000 background coordinates,
227,714 common-negative coordinates and 74,219 eligible Kokoro common-negative
coordinates match exactly. This supports the proposed paired substitution.

Using explicit PCG64 seed90726 and two integer draws per eligible position,
without importing GPU's policy, Mac independently reconstructs all 148,438
draws per word and the complete ordered decision identity:
`484849967208be9ab06ebc2e6fb980f145dc4392f63fb2e3155f6a92bcc05e37`.
Each word selects 37,114 windows: light12,340/time12,347/timer12,427. Every
native request/anchor/voice/speed join and midpoint-quantile fraction matches.
These are prospective decisions, not actual replacement feature tensors.

Ten mutations of the original first batch were rejected: negative start,
wrong crop length/offset/row index, ordinary label mismatch, positive-labeled
prefix, prefix crossing its generated proxy boundary, discarded positive,
wrong batch composition and wrong step. The initial manifest-key preflight
failure (`files` versus actual `payloads`) occurred before metadata loading or
trace reduction; its source identity and exit1 receipt are retained. No
partial output was accepted. The corrected source is bound by all reports.

See `../results/mac-original-broad-saved-window-independent-metadata-audit-v1.json`,
`../results/mac-common-carrier-prospective-decisions-independent-audit-v1.json`,
and the corresponding malformed-control and execution receipts.

## Limits and next step

Ambient source arrays/list lengths and complete original schedule files are
absent on Mac. Local checks validate nonnegative ambient indices and crop
lengths, not source lengths or original tensor hashes. Saved batch hashes are
authenticated values, not locally reconstructed hashes. GPU separately checked
the full original tensor batches. First10k reconstructed batches do not prove
historical live provenance or the whole120k schedule.

The command-carrier coverage gap is real, but its causal role remains unproven.
Two persistent bare errors and the single-Semaine development limitation remain.
New seeded Kokoro realizations also change waveform/duration/crop, so the
intervention is not text-only. Once new features exist, the tensor-schedule
report must describe actual old/new endpoint domains and reachable support
under the frozen discrete midpoint mapping; different lengths need not give
uniform access to every new endpoint.

This closes the saved-coordinate and prospective-choice checks. Continue the
existing material pipeline and paired recognition experiment without repeating
these checks as new inference. No fresh development/acceptance or board/audio
operations occurred; there is no new recognition or P4 qualification result.
