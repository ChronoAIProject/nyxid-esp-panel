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

### Opt-in Hey Nyx preview

`preview-0.6.29-hey-nyx.1` uses the tested original c161 model and the DTLS
session-resource cleanup from PR #15. This release branch builds c161 with
production NyxID endpoints. It has no baked account credentials or local API
configuration. Its `RELEASE_TAG` file makes the normal packaging command emit
manifests with the exact preview download URLs; `VERSION` remains the app
descriptor version `0.6.29`.

Publish this tag as a GitHub **prerelease**, with `latest=false`, only after the
same host tests and both hardware-profile builds succeed. The stable release
workflow is not invoked by `preview-*` tags. Existing panels continue checking
the stable channel and do not install a GitHub prerelease automatically. An
authorized manager can opt in with the existing explicit URL/SHA256 OTA path.
Normal app-only OTA retains saved settings and credentials; factory images are
for initial setup/recovery and overwrite storage. To revert, use an explicit
app-only OTA image from the previous stable release, with its published hash.

Wake sensitivity and nearword generalization remain incomplete. The preview
does not distribute the later failed SpeechT5 candidate or the unscored
MatchboxNet experiment. Neither development nor dining-room panels are updated
as part of publication.

### Stable versions

1. Update `VERSION` and open a PR. Firmware CI runs all host/release tests and
   builds all three hardware profiles with ESP-IDF 5.5.4.
2. Merge the tested change to `main`, then tag that exact commit:

   ```sh
   git tag v0.6.24 <tested-main-commit>
   git push origin v0.6.24
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
