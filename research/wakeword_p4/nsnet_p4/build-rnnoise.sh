#!/usr/bin/env bash
set -eu
: "${IDF_PATH:?set ESP-IDF root}" "${NYX_RNNOISE_SRC:?set RNNoise v0.1.1 checkout}" "${NYX_RNNOISE_WORK:?set build workspace}"
test "$(git -C "$NYX_RNNOISE_SRC" rev-parse HEAD)" = 6cbfd53eb348a8d394e0757b4025c6ded34eb2b6
git -C "$NYX_RNNOISE_SRC" diff --quiet HEAD --
app=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$NYX_RNNOISE_WORK"
cmake -G Ninja -DIDF_TARGET=esp32p4 -DPYTHON="$(command -v python)" \
    -DSDKCONFIG="$NYX_RNNOISE_WORK/sdkconfig" \
    -DSDKCONFIG_DEFAULTS="$app/sdkconfig.defaults" -S "$app" -B "$NYX_RNNOISE_WORK/build"
cmake --build "$NYX_RNNOISE_WORK/build" -j2
