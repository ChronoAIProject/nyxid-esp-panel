# Standalone P4 FP32 resource experiment

This prospective stage runs the **existing unchanged**121596-byte seeded-random
model in the already linked478112-byte numeric probe. It is not a deployment
candidate and accesses no microphone, network or persistent configuration.
The plan binds that exact image/source, the app-only runner and constructed
capture-validator checks. The planner itself never contacts a board.

Before execution, identify the connected development board, obtain fresh app
and active-slot information, and verify the current original-app backup.
Supply the confirmed MAC, LAN status URL, active app offset/slot/size, app ELF
identity and restore binary explicitly. Retain the original partition layout,
NVS, filesystem and OTA metadata. The dining-room production board is excluded.
Follow the repository README's original app-only backup/restore procedure;
do not infer an offset or restore image from a historical run.

```sh
python3 experiments/prepare_fp32_p4_resource_plan_mac_v1.py \
  --repository-root REPO --original-run ORIGINAL_FP32_RUN \
  --constructed-report NEW_CHECK.json --plan NEW_PLAN.json
# Publish these sources/plan before touching hardware. With the actual serial
# and esptool runtime, after the fresh device/slot/backup checks:
PYTHON run_probe.py --fp32-witness \
  --port DEV_PORT --expected-mac VERIFIED_DEV_MAC --status-url CURRENT_STATUS_URL \
  --restore-app CURRENT_BACKUP --expected-app-sha256 CURRENT_APP_ELF_SHA \
  --app-offset VERIFIED_OFFSET --slot VERIFIED_ACTIVE_SLOT --slot-bytes VERIFIED_SIZE \
  --probe-bin ORIGINAL_FP32_RUN/build/nyx_fp32_feasibility.bin \
  --expected-probe-sha256 FROZEN_PROBE_SHA \
  --vectors-manifest ORIGINAL_FP32_RUN/export-result.json --output NEW_PRIVATE_RUN
```

The runner preserves its existing preflight and finally-restoration behavior.
The added FP32 mode requires exactly eight unique condition/repeat cases and a
successful terminal marker, finite outputs, maximum absolute error<=1e-4 and
every measured Invoke<=30000us. Cold calls remain included. It captures main
arena usage and the64KiB variable arena capacity, which is an upper bound, not
measured variable usage. The source has256 calls per case; the on-device report
does not dump individual output samples or prove repeat byte equality.

A missed deadline or numeric/allocation failure is still reported and the app
is restored. Keep original private UART/flash logs and filtered public numeric
results separate; independently verify the original app, slot, service and
wake state after restoration. Passing this standalone hop deadline cannot
establish full-panel headroom or trained-model/recognition acceptance.

The capture validator's constructed checks cover exact limit equality,
missing/duplicate cases, failed markers, NaN/infinity, negative/zero/overflowed
measurements, malformed boolean identifiers/arenas, missing fields and missed
hop deadlines. These are preparation evidence, not real P4 observations.
