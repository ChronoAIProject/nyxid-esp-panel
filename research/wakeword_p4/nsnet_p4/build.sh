#!/usr/bin/env bash
set -eu
: "${IDF_PATH:?set ESP-IDF root}" "${NYX_SR_COMPONENT:?set ESP-SR component path}" "${NYX_NSNET_WORK:?set build workspace}"
app=$(cd -- "$(dirname -- "$0")" && pwd)
unset NYX_RNNOISE_SRC
managed=$(dirname -- "$NYX_SR_COMPONENT")
mkdir -p "$NYX_NSNET_WORK/models"
defaults="$app/sdkconfig.defaults"
if [[ ${NYX_NSNET_BOARD:-OFF} == ON ]]; then
    defaults="$defaults;$app/sdkconfig.board.defaults"
fi
cp -r "$NYX_SR_COMPONENT/model/nsnet_model/nsnet2" "$NYX_NSNET_WORK/models/"
export NYX_NSNET_PACK="$NYX_NSNET_WORK/models.bin"
python3 "$NYX_SR_COMPONENT/model/pack_model.py" -m "$NYX_NSNET_WORK/models" -o "$NYX_NSNET_PACK"
cmake -G Ninja -DIDF_TARGET=esp32p4 -DPYTHON="$(command -v python)" \
    -DEXTRA_COMPONENT_DIRS="$managed/espressif__dl_fft;$managed/espressif__cjson" \
    -DSDKCONFIG="$NYX_NSNET_WORK/sdkconfig" \
    -DNYX_NSNET_BOARD="${NYX_NSNET_BOARD:-OFF}" -DNYX_NSNET_ANSI_FFT="${NYX_NSNET_ANSI_FFT:-OFF}" \
    -DSDKCONFIG_DEFAULTS="$defaults" \
    -S "$app" -B "$NYX_NSNET_WORK/build"
cmake --build "$NYX_NSNET_WORK/build" -j2
