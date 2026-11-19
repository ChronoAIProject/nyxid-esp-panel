#!/usr/bin/env bash
# Materialize an isolated research listener. Only its microphone matcher changes.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
base=419837340caeb739c71834d7a2350cc2b39f1de3
app=research/wakeword_p4/first_command_p4/mac_listener
export NYX_MAC_WORK="$root/.local/wakeword/c161-mac-listener"
export NYX_SR_COMPONENT="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__esp-sr"
export IDF_TOOLS_PATH="$root/.local/wakeword/idf-tools"
export PATH="/usr/bin:$PATH"
work="$NYX_MAC_WORK/final-marker-confirmation"
src="$work/src"
mkdir -p "$src"
git archive "$base" "$app/main" "$app/CMakeLists.txt" "$app/dependencies.lock" research/wakeword_p4/portable research/wakeword_p4/first_command_p4/main components/panel_services | tar -x -C "$src"
(
    source .local/wakeword/c161-onset-names/env
    PYTHONPATH="$app:$PYTHONPATH" /usr/bin/python3 - "$src/$app/main" <<'PY'
from pathlib import Path
import sys
from check_final_marker_confirmation import candidate_header
main = Path(sys.argv[1])
(main / "final_marker.h").write_text(candidate_header())
path = main / "listener.cpp"
source = path.read_text()
original = "nyx::CodedCueV5 cue;"
assert source.count(original) == 1
path.write_text('#include "final_marker.h"\n' + source.replace(original, "nyx::FinalMarkerConfirmation cue;"))
PY
)
if [[ ! -e "$src/$app/managed_components" ]]; then
    ln -s "$(realpath "$app/managed_components")" "$src/$app/managed_components"
fi
cp "$NYX_MAC_WORK/sdkconfig" "$work/sdkconfig"
source .local/wakeword/esp-idf/export.sh > "$work/environment.log" 2>&1
fft="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__dl_fft"
idf.py -C "$src/$app" -B "$work/build" -DIDF_TARGET=esp32p4 "-DEXTRA_COMPONENT_DIRS=$fft;$NYX_MAC_WORK/features" \
    "-DSDKCONFIG=$work/sdkconfig" "-DSDKCONFIG_DEFAULTS=$root/research/wakeword_p4/midlevel_p4_selftest/sdkconfig.defaults" reconfigure
cmake --build "$work/build" --parallel 2
