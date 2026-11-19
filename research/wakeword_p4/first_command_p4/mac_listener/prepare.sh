#!/usr/bin/env bash
# Run from the research checkout. Use the exact runtime and frontend used by Mac.
set -euo pipefail
work="$PWD/.local/wakeword/c161-mac-listener"
commit=0d8e1935702a561b535a8c1a0921fd8034943d2c
mkdir -p "$work/src" "$work/features"
git archive "$commit" components/panel_wake_runtime models/hey-nyx-c161 patches/micro_speech_filterbank_allocation.patch | tar -x -C "$work/src"
zip="$work/features.zip"
if [[ ! -f "$zip" ]]; then
    curl -qfsSL --retry 2 https://components-file.espressif.com/components/esphome/esp-micro-speech-features/1.2.3/esphome__esp-micro-speech-features-v1.2.3.zip -o "$zip.tmp"
    mv "$zip.tmp" "$zip"
fi
printf '%s  %s\n' 052a8f32e7a82f49593bfdc067638a8cee6542443726d1a7e5b01b3bd9d67eac "$zip" | sha256sum -c -
unzip -oq "$zip" -d "$work/features"
git apply --directory=.local/wakeword/c161-mac-listener/features -p2 "$work/src/patches/micro_speech_filterbank_allocation.patch"
python3 "$work/src/components/panel_wake_runtime/prepare_streaming_runtime.py" \
    --model "$work/src/models/hey-nyx-c161/model.tflite" --profile "$work/src/models/hey-nyx-c161/profile.json" --output "$work/model"
