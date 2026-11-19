#!/usr/bin/env bash
set -euo pipefail
root="$PWD"
fixtures="$root/.local/wakeword/c161-first-command-v5"
work="$root/.local/wakeword/c161-opus-rate-comparison"
project="$work/src/research/wakeword_p4/first_command_p4"
export IDF_PATH="$root/.local/wakeword/esp-idf"
export IDF_TOOLS_PATH="$root/.local/wakeword/idf-tools"
export IDF_PYTHON_ENV_PATH="$IDF_TOOLS_PATH/python_env/idf5.5_py3.11_env"
export PATH="$IDF_PYTHON_ENV_PATH/bin:$IDF_TOOLS_PATH/tools/riscv32-esp-elf/esp-14.2.0_20260121/riscv32-esp-elf/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=2
mkdir -p "$project" "$work/src/components"
cp -r research/wakeword_p4/first_command_p4/{CMakeLists.txt,main} "$project/"
ln -sfn "$root/research/wakeword_p4/portable" "$project/../portable"
ln -sfn "$root/research/wakeword_p4/results" "$project/../results"
ln -sfn "$root/components/panel_services" "$work/src/components/panel_services"
cp "$fixtures/sdkconfig" "$work/sdkconfig"
cmake -G Ninja -S "$project" -B "$work/build" -DIDF_TARGET=esp32p4 -DSDKCONFIG="$work/sdkconfig" \
    -DPYTHON="$IDF_PYTHON_ENV_PATH/bin/python" -DEXTRA_COMPONENT_DIRS="$fixtures/esp_audio_codec" > "$work/configure.log" 2>&1
ninja -C "$work/build" -j2 > "$work/build.log" 2>&1
