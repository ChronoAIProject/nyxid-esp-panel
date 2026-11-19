#!/usr/bin/env bash
# Build a separate experimental listener against the pinned framed listener source.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
base=b65b73659e0bceb3205788d7052bd43597dbe08e
app=research/wakeword_p4/first_command_p4/mac_listener
export NYX_MAC_WORK="$root/.local/wakeword/c161-mac-listener"
export NYX_SR_COMPONENT="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__esp-sr"
export IDF_TOOLS_PATH="$root/.local/wakeword/idf-tools"
export PATH="/usr/bin:$PATH"
work="$NYX_MAC_WORK/first-reference-wait"
src="$work/src"
mkdir -p "$src"
git archive "$base" "$app" research/wakeword_p4/portable research/wakeword_p4/first_command_p4/main components/panel_services | tar -x -C "$src"
patch -s -d "$src" -p1 < "$app/first_reference_wait.patch"
source .local/wakeword/esp-idf/export.sh > "$work/environment.log" 2>&1
fft="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__dl_fft"
managed="$root/$app/managed_components"
if [[ -d "$managed" && ! -e "$src/$app/managed_components" ]]; then
    ln -s "$(realpath "$managed")" "$src/$app/managed_components"
fi
cp "$NYX_MAC_WORK/sdkconfig" "$work/sdkconfig"
idf.py -C "$src/$app" -B "$work/build" -DIDF_TARGET=esp32p4 "-DEXTRA_COMPONENT_DIRS=$fft;$NYX_MAC_WORK/features" \
    "-DSDKCONFIG=$work/sdkconfig" "-DSDKCONFIG_DEFAULTS=$root/research/wakeword_p4/midlevel_p4_selftest/sdkconfig.defaults" reconfigure
cmake --build "$work/build" --parallel 2
