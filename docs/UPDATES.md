# Firmware releases and automatic OTA

Production panels check the stable GitHub Release channel one minute after boot
and every six hours afterwards. Failed checks retry after 15 minutes. No Mac,
NyxID deployment, API token or user account is needed for release downloads.
Network outages leave the installed firmware running.

Updates wait for an idle microphone and camera. Voice calls cannot start during
OTA. The app image enters the existing inactive A/B slot; Wi-Fi, NyxID keys,
voice settings and the saved default picture remain in their existing partitions.
Do not disconnect power while the update is being written.

Only a strictly newer stable `major.minor.patch` version for the compiled
hardware profile is eligible. Legacy P4 and revision 3 P4 have separate release
assets and manifests. The integrated LCD-X8 board also uses its own
`p4-rev3-lcd-x8-16mb` assets, so it cannot automatically install an external-screen
image. HTTPS certificate validation stays enabled across every
redirect; release URLs are restricted to this repository and GitHub asset hosts.
The manifest's exact size, SHA256, project name and app-descriptor version are
checked before flash writes. IDF validates image/chip compatibility before
selecting the new boot slot. The HTTPS GitHub repository is the publication trust
source; these manifests do not introduce an independent offline signing key.

The existing 30-second startup health check validates a new image or rolls back.
The target version is saved before changing the boot slot. If the board returns
to an older version, that failed target is blocked from automatic reinstallation,
including after power cycles. A subsequent newer release is still eligible.
Explicit SHA256 URL/raw OTA remains available to an authorized manager for recovery.
There is no automatic partition-table or bootloader migration.

Firmware compiled with the local Agent API development profile reports
`development` and disables this production update channel, including manual
release-channel installation. Its existing explicit LAN OTA recovery stays usable.

## Remote configuration

Only the panel's **management service** can call `/system/update`. The daily
service can read the same update summary under `/status.update`, but cannot
configure or initiate updates.

```sh
nyxid proxy request "$PANEL_ADMIN_SLUG" /system/update --output json
nyxid proxy request "$PANEL_ADMIN_SLUG" /system/update -m POST \
  -H Content-Type:application/json -d '{"automatic":false}'
nyxid proxy request "$PANEL_ADMIN_SLUG" /system/update -m POST \
  -H Content-Type:application/json -d '{"action":"check"}'
nyxid proxy request "$PANEL_ADMIN_SLUG" /system/update -m POST \
  -H Content-Type:application/json -d '{"action":"install"}'
```

POST takes exactly one field. `check` only checks; `install` checks and installs
an eligible newer release once idle, even with automatic updates disabled.
Disabling automatic updates cancels waiting automatic installations; an image
already being written finishes. `202` acknowledges queued work. Poll `/status`:
`update` reports release checks, available/latest/blocked versions and the saved
switch; `ota` reports actual download/write/reboot errors and the running slot.
Settings → Device status shows update state and offers a check button.
USB console equivalent: `panel-update status|check|install|auto on|off`.

## Publish a version

1. Update `VERSION` and open a PR. Firmware CI runs all host/release tests and
   builds all three hardware profiles with ESP-IDF 5.5.4.
2. Merge the tested change to `main`, then tag that exact commit:

   ```sh
   git tag v0.6.29 <tested-main-commit>
   git push origin v0.6.29
   ```

3. The **Release firmware** workflow verifies the stable tag matches `VERSION`
   and belongs to `main`, reruns the same build/test workflow, checks all asset
   hashes, uploads to a draft, then publishes it as latest. Published assets
   are never overwritten. A failed publication leaves no partial public update;
   remove an incomplete draft before rerunning. The workflow also accepts an
   existing tag through manual dispatch.

Each release includes, per profile:

- `nyxid-esp-panel-<profile>.bin`: application-only OTA image.
- `update-<profile>.json`: bounded machine-readable release identity and hash.
- `nyxid-esp-panel-<profile>-factory.zip`: whitelisted factory flash files.
- `checksums-<profile>.txt`: SHA256 for those three assets.

Factory flash is for initial setup/recovery and can overwrite existing storage.
Normal updates use only the app binary. Existing v0.6.23 and earlier boards need
one explicit OTA or USB installation of this updater; subsequent releases update
without the development machine.

## v1.0.0

All three hardware profiles ship the C161 “Hey Nyx” wake word. This release
includes a fix for low-memory startup of the audio frontend. If its filterbank
work buffer cannot be allocated, initialization returns an error so the caller
can release partially allocated memory.

The C161 model and settings match v0.6.29. On the 10.1-inch rev1 development
panel, a speaker woke the device and gave a Mandarin command in one utterance.
The voice service received the full command and the panel repeated the requested
phrase. In separate speaker playback across three synthetic voices, the panel
woke on 5 of 12 attempts and preserved the complete command on all five.
A separate streaming test found a false wake on “Hey Nicole”.
[Rev1 voice results](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7041028477)
· [False-wake results](https://github.com/ChronoAIProject/nyxid-esp-panel/issues/18#issuecomment-7041138575)

Production panels running v0.6.29 can receive v1.0.0 through the stable OTA
channel. For a manual upgrade, use the app image matching the hardware profile.

## v0.6.29

The stable channel uses Hey Nyx C161 on all three hardware profiles and includes
the board isolation from PR #17. Existing production credentials and storage
are preserved. LCD-X8 has separate audio/display wiring and release assets.
C161 sensitivity remains under improvement; this release uses the original
model and deployed settings, without training or audio-speed changes.

The earlier `preview-0.6.29-hey-nyx.1` also reports application version 0.6.29.
Those devices require explicit app-only OTA or USB installation of the stable
image; the automatic updater only accepts a strictly newer application version.

## Branches and channels

- `main` is the default stable branch. Merge approved/tested changes here and
  tag `vX.Y.Z` matching `VERSION` to publish a formal release and update stable OTA.
- `dev` is the integration/test branch. Each push runs host/model/release tests
  and all board builds, then publishes an immutable `dev-vX.Y.Z-RUN-SHA`
  prerelease. Its manifests point to that exact release. It never becomes
  GitHub's latest stable release, and normal panels do not auto-install it.
- Feature/fix branches target `dev`; promote tested `dev` changes to `main` by PR.
  Keep `dev` synchronized with `main` after a stable release. Research branches
  stay separate until a candidate is selected and tested.

The C161 detector, voice integration and model preparation source are preserved
unchanged from the existing tested implementation. See the
[source receipt](evidence/2026-11-17-c161-source.json). Production builds select
production endpoints rather than embedding the monitored local server address.
