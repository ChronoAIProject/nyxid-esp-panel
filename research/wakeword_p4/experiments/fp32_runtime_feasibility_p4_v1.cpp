// Candidate-free synthetic numeric probe. No sensor, network, NVS or flashing code.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "expected.h"

extern const uint8_t model[] asm("_binary_model_tflite_start");

static bool run(unsigned condition, unsigned repeat) {
    constexpr size_t arena_size = 256 * 1024, variable_size = 64 * 1024;
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    auto *variable = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, variable_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena || !variable) {
        printf("FP32_ALLOCATION_FAILED {\"condition\":%u,\"repeat\":%u}\n", condition, repeat);
        free(arena); free(variable); return false;
    }
    bool passed = false;
    {
        auto *allocator = tflite::MicroAllocator::Create(variable, variable_size);
        auto *variables = tflite::MicroResourceVariables::Create(allocator, 20);
        tflite::MicroMutableOpResolver<20> ops;
        ops.AddCallOnce(); ops.AddVarHandle(); ops.AddReadVariable(); ops.AddReshape();
        ops.AddConcatenation(); ops.AddStridedSlice(); ops.AddAssignVariable(); ops.AddConv2D();
        ops.AddDepthwiseConv2D(); ops.AddSplitV(); ops.AddFullyConnected(); ops.AddLogistic();
        ops.AddExpandDims(); ops.AddMul(); ops.AddAdd(); ops.AddMinimum(); ops.AddMaximum();
        tflite::MicroInterpreter interpreter(tflite::GetModel(model), ops, arena, arena_size, variables);
        if (interpreter.AllocateTensors() == kTfLiteOk) {
            auto *input = interpreter.input(0); auto *output = interpreter.output(0);
            if (input->type == kTfLiteFloat32 && input->bytes == 480 && output->type == kTfLiteFloat32 && output->bytes == 4) {
                uint32_t state = 0x4e595831;
                int64_t durations[256]{};
                float max_error = 0;
                passed = true;
                for (unsigned step = 0; step < 256; ++step) {
                    for (unsigned j = 0; j < 120; ++j) {
                        float value = 0;
                        if (condition == 1) value = 26;
                        if (condition == 2) value = float((step * 120 + j) % 257) * (26.f / 256.f);
                        if (condition == 3) {
                            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                            value = float(state & 65535) * (26.f / 65535.f);
                        }
                        input->data.f[j] = value;
                    }
                    auto start = esp_timer_get_time();
                    if (interpreter.Invoke() != kTfLiteOk) { passed = false; break; }
                    durations[step] = esp_timer_get_time() - start;
                    float actual = output->data.f[0];
                    if (!std::isfinite(actual)) { passed = false; break; }
                    max_error = std::max(max_error, std::fabs(actual - expected[condition][step]));
                }
                std::sort(durations, durations + 256);
                passed = passed && max_error <= 1e-4f;
                printf("FP32_CASE {\"condition\":%u,\"repeat\":%u,\"passed\":%s,\"max_abs_error\":%.9g,\"arena_used\":%u,\"variable_arena_allocated\":%u,\"p50_us\":%lld,\"p95_us\":%lld,\"max_us\":%lld}\n",
                       condition, repeat, passed ? "true" : "false", double(max_error),
                       unsigned(interpreter.arena_used_bytes()), unsigned(variable_size),
                       (long long)durations[128], (long long)durations[243], (long long)durations[255]);
            }
        }
    }
    free(arena); free(variable); return passed;
}

extern "C" void app_main() {
    bool passed = true;
    for (unsigned condition = 0; condition < 4; ++condition)
        for (unsigned repeat = 0; repeat < 2; ++repeat)
            passed = run(condition, repeat) && passed;
    printf("FP32_SYNTHETIC_DONE {\"passed\":%s,\"stack_free\":%u}\n", passed ? "true" : "false", unsigned(uxTaskGetStackHighWaterMark(nullptr)));
}
