#!/usr/bin/env bash
# Compile the panel-only reference pair beside the candidate microphone matcher.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
app=research/wakeword_p4/first_command_p4/mac_listener
work="$root/.local/wakeword/c161-panel-marker-order"
export NYX_MAC_WORK="$root/.local/wakeword/c161-mac-listener"
export NYX_SR_COMPONENT="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__esp-sr"
export IDF_TOOLS_PATH="$root/.local/wakeword/idf-tools"
export PATH="/usr/bin:$PATH"
mkdir -p "$work/src"
git archive 419837340caeb739c71834d7a2350cc2b39f1de3 "$app/main" "$app/CMakeLists.txt" "$app/dependencies.lock" research/wakeword_p4/portable research/wakeword_p4/first_command_p4/main components/panel_services | tar -x -C "$work/src"
(
    source .local/wakeword/c161-onset-names/env
    PYTHONPATH="$app:$PYTHONPATH" /usr/bin/python3 - "$work/src/$app/main" <<'PY'
from pathlib import Path
import sys
from check_panel_marker_order import host, reference_header, VETO
main = Path(sys.argv[1])
(main/"final_marker.h").write_text(host.candidate_header())
(main/"panel_marker_pair.h").write_text(reference_header())
p = main/"listener.cpp"
s = p.read_text()
edits = {
    'nyx::CodedCueV5 cue;': 'nyx::FinalMarkerConfirmation cue;\n    nyx::PanelMarkerPair panel_pair;\n    int panel_remaining = 0;',
    'const bool full_clear =': f'if (panel_pair.update(reference)) panel_remaining = {VETO} + 1;\n            const bool panel_clear = panel_remaining == 0;\n            if (panel_remaining) --panel_remaining;\n            const bool full_clear =',
    'model_route.update_active(active,reference_clear,model_hit': 'model_route.update_active(active,reference_clear && panel_clear,model_hit',
}
for old, new in edits.items():
    assert s.count(old) == 1
    s = s.replace(old,new)
p.write_text('#include "panel_marker_pair.h"\n#include "final_marker.h"\n' + s)
PY
)
if [[ ! -e "$work/src/$app/managed_components" ]]; then
    ln -s "$(realpath "$app/managed_components")" "$work/src/$app/managed_components"
fi
cp "$NYX_MAC_WORK/sdkconfig" "$work/sdkconfig"
source .local/wakeword/esp-idf/export.sh > "$work/environment.log" 2>&1
fft="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__dl_fft"
idf.py -C "$work/src/$app" -B "$work/build" -DIDF_TARGET=esp32p4 "-DEXTRA_COMPONENT_DIRS=$fft;$NYX_MAC_WORK/features" \
    "-DSDKCONFIG=$work/sdkconfig" "-DSDKCONFIG_DEFAULTS=$root/research/wakeword_p4/midlevel_p4_selftest/sdkconfig.defaults" reconfigure
cmake --build "$work/build" --parallel 2
