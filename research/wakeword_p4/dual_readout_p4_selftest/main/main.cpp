#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern const uint8_t dual_model[] asm("_binary_4090_c161_dual_readout_v1_tflite_start");
extern const uint8_t fixture[] asm("_binary_fixture_bin_start");
extern const uint8_t fixture_end[] asm("_binary_fixture_bin_end");

extern "C" void app_main() {
    alignas(16) static uint8_t tensor_arena[65536];
    alignas(16) static uint8_t variable_arena[16384];
    auto* allocator = tflite::MicroAllocator::Create(variable_arena, sizeof(variable_arena));
    if (!allocator) { std::puts("NYX_DUAL_FAIL allocator"); return; }
    auto* variables = tflite::MicroResourceVariables::Create(allocator, 24);
    if (!variables) { std::puts("NYX_DUAL_FAIL variables"); return; }
    tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddCallOnce(); resolver.AddVarHandle(); resolver.AddReadVariable(); resolver.AddReshape();
    resolver.AddConcatenation(); resolver.AddStridedSlice(); resolver.AddAssignVariable(); resolver.AddConv2D();
    resolver.AddDepthwiseConv2D(); resolver.AddSplitV(); resolver.AddFullyConnected(); resolver.AddLogistic();
    resolver.AddQuantize(); resolver.AddExpandDims();
    tflite::MicroInterpreter interpreter(tflite::GetModel(dual_model), resolver,
                                        tensor_arena, sizeof(tensor_arena), variables);
    if (interpreter.AllocateTensors() != kTfLiteOk) { std::puts("NYX_DUAL_FAIL allocation"); return; }
    auto* in = interpreter.input(0);
    auto* base = interpreter.output(0);
    auto* trained = interpreter.output(1);
    constexpr int stride = 122;
    const int length = fixture_end - fixture;
    if (length <= 0 || length % stride || !in || in->bytes != 120 || in->type != kTfLiteInt8 ||
        !base || !trained || base->bytes != 1 || trained->bytes != 1 ||
        base->type != kTfLiteUInt8 || trained->type != kTfLiteUInt8) {
        std::puts("NYX_DUAL_FAIL io"); return;
    }
    int errors[2]{}, first = -1, differing = 0;
    int64_t slowest = 0;
    for (int i = 0; i < length / stride; ++i) {
        const auto* row = fixture + i * stride;
        std::memcpy(in->data.int8, row, 120);
        const auto start = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) { std::puts("NYX_DUAL_FAIL invoke"); return; }
        slowest = std::max(slowest, esp_timer_get_time() - start);
        errors[0] += base->data.uint8[0] != row[120];
        errors[1] += trained->data.uint8[0] != row[121];
        differing += row[120] != row[121];
        if (first < 0 && (errors[0] || errors[1])) first = i;
    }
    std::printf("NYX_DUAL_%s calls=%d distinct=%d errors=%d,%d first=%d arena=%u var_alloc=%u max_us=%lld heap_min=%u\n",
                first < 0 && differing ? "PASS" : "FAIL", length / stride, differing, errors[0], errors[1], first,
                unsigned(interpreter.arena_used_bytes()), unsigned(sizeof(variable_arena)), (long long)slowest,
                unsigned(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
}
