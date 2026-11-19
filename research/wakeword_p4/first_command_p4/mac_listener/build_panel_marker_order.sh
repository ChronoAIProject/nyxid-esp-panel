#!/usr/bin/env bash
# Compile the panel-only reference pair beside the candidate microphone matcher.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
app=research/wakeword_p4/first_command_p4/mac_listener
variant=${1:-original}
case "$variant" in
    original) work="$root/.local/wakeword/c161-panel-marker-order" ;;
    timed) work="$root/.local/wakeword/c161-panel-marker-order-timed" ;;
    tx) work="$root/.local/wakeword/c161-panel-playout-event" ;;
    echo) work="$root/.local/wakeword/c161-panel-echo-confirmation" ;;
    middle) work="$root/.local/wakeword/c161-panel-middle-confirmation" ;;
    *) exit 2 ;;
esac
export NYX_MAC_WORK="$root/.local/wakeword/c161-mac-listener"
export NYX_SR_COMPONENT="$root/.local/wakeword/hey-reference-qemu-project-v2/managed_components/espressif__esp-sr"
export IDF_TOOLS_PATH="$root/.local/wakeword/idf-tools"
export PATH="/usr/bin:$PATH"
mkdir -p "$work/src"
git archive 419837340caeb739c71834d7a2350cc2b39f1de3 "$app/main" "$app/CMakeLists.txt" "$app/dependencies.lock" research/wakeword_p4/portable research/wakeword_p4/first_command_p4/main components/panel_services | tar -x -C "$work/src"
(
    source .local/wakeword/c161-onset-names/env
    PYTHONPATH="$app:$PYTHONPATH" /usr/bin/python3 - "$work/src/$app/main" "$variant" <<'PY'
from pathlib import Path
import sys
from check_panel_marker_order import host, reference_header, VETO
main = Path(sys.argv[1])
timed = sys.argv[2] in ("timed","tx","echo","middle")
matcher = "TimedFinalMarker" if timed else "FinalMarkerConfirmation"
header = host.HEADER.with_name("c161_timed_final_marker.h").read_text() if timed else host.candidate_header()
(main/"final_marker.h").write_text(header)
(main/"panel_marker_pair.h").write_text(reference_header())
p = main/"listener.cpp"
s = p.read_text()
edits = {
    'nyx::CodedCueV5 cue;': f'nyx::{matcher} cue;\n    nyx::PanelMarkerPair panel_pair;\n    int panel_remaining = 0;',
    'const bool full_clear =': f'if (panel_pair.update(reference)) panel_remaining = {VETO} + 1;\n            const bool panel_clear = panel_remaining == 0;\n            if (panel_remaining) --panel_remaining;\n            const bool full_clear =',
    'model_route.update_active(active,reference_clear,model_hit': 'model_route.update_active(active,reference_clear && panel_clear,model_hit',
}
if sys.argv[2] in ("tx","echo","middle"):
    header = host.HEADER.with_name("c161_panel_demo_guard.h")
    (main/header.name).write_bytes(header.read_bytes())
    edits["nyx::CodedCueV5 cue;"] = f"nyx::{matcher} cue;\n    nyx::PanelMarkerPair panel_pair;\n    nyx::PanelDemoGuard panel_guard;"
    edits["const bool full_clear ="] = "const bool panel_clear = panel_guard.allow_model(panel_pair.update(reference), played_pair == uint32_t(at+i+1));\n            const bool full_clear ="
    edits["const bool qualify_capture_frame = h[0] == 0x4652594e;"] = (
        "const bool event_fixture = h[0] == 0x5854594e;\n    const bool qualify_capture_frame = event_fixture || h[0] == 0x4652594e;")
    edits["const auto& wn="] = (
        "uint32_t played_pair = 0;\n    if (event_fixture) ESP_ERROR_CHECK(esp_flash_read(nullptr,&played_pair,kOffset+sizeof(h)+length*4,sizeof(played_pair)));\n    const auto& wn=")
    s = '#include "c161_panel_demo_guard.h"\n' + s
if sys.argv[2] in ("echo","middle"):
    header = host.HEADER.with_name("c161_panel_echo_confirmation.h")
    (main/header.name).write_bytes(header.read_bytes())
    s = '#include "c161_panel_echo_confirmation.h"\n' + s
    edits["nyx::CodedCueV5 cue;"] = f"nyx::{matcher} cue;\n    nyx::PanelMarkerPair panel_pair, mic_pair;\n    nyx::PanelEchoConfirmation panel_guard;"
    edits["const bool full_clear ="] = (
        "const bool panel_clear = panel_guard.allow_model(panel_pair.update(reference), "
        "played_pair == uint32_t(at+i+1), mic_pair.update(pcm[at+i]));\n            const bool full_clear =")
if sys.argv[2] == "middle":
    header = host.HEADER.with_name("c161_panel_middle_marker.h")
    (main/header.name).write_bytes(header.read_bytes())
    s = '#include "c161_panel_middle_marker.h"\n' + s
    edits["nyx::CodedCueV5 cue;"] = f"nyx::{matcher} cue;\n    nyx::PanelMarkerPair panel_pair;\n    nyx::PanelMiddleMarker mic_pair;\n    nyx::PanelEchoConfirmation panel_guard;"
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
