#include <cstdint>
#include <cstdio>

#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern const uint8_t model_bytes[] asm("_binary_4090_c161_onset_evidence_posterior_int8_tflite_start");

namespace {
constexpr uint32_t fixture_offset = 2 << 20;
constexpr uint32_t feature_bytes = 144 * 40;
constexpr uint32_t classes = 5;
alignas(16) uint8_t arena[192 * 1024];
tflite::MicroMutableOpResolver<7> resolver;

bool replay() {
    resolver.AddPad();
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    const auto start = esp_timer_get_time();
    tflite::MicroInterpreter vm(tflite::GetModel(model_bytes), resolver, arena, sizeof(arena));
    if (vm.AllocateTensors() != kTfLiteOk) return false;
    auto* input = vm.input(0);
    auto* output = vm.output(0);
    if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        input->bytes != feature_bytes || output->bytes != classes) return false;
    std::printf("ONSET_READY arena=%u init_us=%lld\n", (unsigned)vm.arena_used_bytes(), esp_timer_get_time()-start);
    uint32_t count;
    if (esp_flash_read(nullptr, &count, fixture_offset, sizeof(count)) != ESP_OK || count == 0 || count > 300) return false;
    uint32_t address = fixture_offset + sizeof(count);
    for (unsigned i = 0; i < count; ++i) {
        if (esp_flash_read(nullptr, input->data.int8, address, feature_bytes) != ESP_OK) return false;
        address += feature_bytes + classes;
        const auto begin = esp_timer_get_time();
        if (vm.Invoke() != kTfLiteOk) return false;
        const auto elapsed = esp_timer_get_time()-begin;
        const auto* y = output->data.int8;
        std::printf("ONSET_ROW %u %d %d %d %d %d %lld\n", i, y[0], y[1], y[2], y[3], y[4], elapsed);
        vTaskDelay(1);
    }
    std::printf("ONSET_END count=%u min_internal=%u\n", (unsigned)count,
                (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
    return true;
}
}

extern "C" void app_main() {
    if (!replay()) std::puts("ONSET_FAIL");
    std::puts("ONSET_DONE");
}
