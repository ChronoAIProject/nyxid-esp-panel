// Numeric/resource-only development probe. No microphone, network or NVS writes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "reference_fc.h"
#ifdef NYX_HEAD_PROBE_TRACE
#include "trace.h"
#endif

extern const uint8_t model[] asm("_binary_head_tflite_start");
extern const uint8_t model_end[] asm("_binary_head_tflite_end");
extern const int8_t inputs[] asm("_binary_inputs_bin_start");
extern const int8_t inputs_end[] asm("_binary_inputs_bin_end");
extern const int8_t expected[] asm("_binary_expected_bin_start");
extern const int8_t expected_end[] asm("_binary_expected_bin_end");

static constexpr size_t input_values = 16 * 96;
static constexpr size_t arena_bytes = 64 * 1024;

static bool benchmark(const char *name, uint32_t caps, const uint8_t *data) {
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, arena_bytes, caps | MALLOC_CAP_8BIT));
    if (!arena) {
        printf("PROBE_HEAD {\"name\":\"%s\",\"allocated\":false}\n", name);
        return false;
    }
    bool passed = false;
    {
        tflite::MicroMutableOpResolver<14> ops;
        ops.AddReshape(); ops.AddFullyConnected(nyx_head_probe::reference_fc_registration()); ops.AddMean();
        ops.AddDequantize(); ops.AddNeg(); ops.AddQuantize();
        ops.AddSquaredDifference(); ops.AddAdd(); ops.AddRsqrt();
        ops.AddMul(); ops.AddLogistic();
        ops.AddConv2D(); ops.AddMaxPool2D(); ops.AddReduceMax();
#ifdef NYX_HEAD_PROBE_TRACE
        nyx_head_probe::trace_registrations(ops);
#endif
        tflite::MicroInterpreter interpreter(tflite::GetModel(data), ops, arena, arena_bytes);
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            printf("PROBE_HEAD {\"name\":\"%s\",\"tensors\":false}\n", name);
        } else {
            auto *in = interpreter.input(0), *out = interpreter.output(0);
            size_t cases = size_t(inputs_end - inputs) / input_values;
            bool shape_ok = in->type == kTfLiteInt8 && in->bytes == input_values &&
                out->type == kTfLiteInt8 && out->bytes == 1 &&
                in->params.scale == 1.f / 128 && in->params.zero_point == 0 &&
                out->params.scale == 1.f / 256 && out->params.zero_point == -128 &&
                size_t(inputs_end - inputs) == cases * input_values &&
                size_t(expected_end - expected) == cases && cases > 0;
            if (!shape_ok) {
                printf("PROBE_HEAD {\"name\":\"%s\",\"shape_ok\":false}\n", name);
            } else {
                auto *times = static_cast<int64_t *>(malloc(cases * sizeof(int64_t)));
                if (times) {
                    unsigned done = 0, mismatches = 0, max_delta = 0;
                    for (size_t n = 0; n < cases; ++n) {
                        memcpy(in->data.int8, inputs + n * input_values, input_values);
#ifdef NYX_HEAD_PROBE_TRACE
                        nyx_head_probe::tracing = n == 269;
#endif
                        int64_t start = esp_timer_get_time();
                        if (interpreter.Invoke() != kTfLiteOk) break;
                        times[done++] = esp_timer_get_time() - start;
                        unsigned delta = unsigned(std::abs(int(out->data.int8[0]) - int(expected[n])));
                        if (delta && mismatches < 8)
                            printf("HEAD_MISMATCH case=%u actual=%d expected=%d\n",
                                unsigned(n), int(out->data.int8[0]), int(expected[n]));
                        mismatches += delta > 0;
                        max_delta = std::max(max_delta, delta);
                        vTaskDelay(pdMS_TO_TICKS(10));
                    }
                    std::sort(times, times + done);
                    passed = done == cases && mismatches == 0;
                    printf("PROBE_HEAD {\"name\":\"%s\",\"calls\":%u,\"arena_allocated\":%u,"
                        "\"arena_used\":%u,\"p50_us\":%lld,\"p95_us\":%lld,\"max_us\":%lld,"
                        "\"mismatches\":%u,\"max_output_delta\":%u,\"passed\":%s}\n",
                        name, done, unsigned(arena_bytes), unsigned(interpreter.arena_used_bytes()),
                        (long long)(done ? times[done / 2] : 0),
                        (long long)(done ? times[done * 95 / 100] : 0),
                        (long long)(done ? times[done - 1] : 0), mismatches, max_delta,
                        passed ? "true" : "false");
                    free(times);
                } else printf("PROBE_HEAD {\"name\":\"%s\",\"times_allocated\":false}\n", name);
            }
        }
    }
    free(arena);
    fflush(stdout);
    return passed;
}

extern "C" void app_main() {
    printf("HEAD_PROBE_START internal_largest=%u psram_largest=%u\n",
        unsigned(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        unsigned(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
    bool passed = benchmark("internal64k_model_flash", MALLOC_CAP_INTERNAL, model);
    const size_t size = size_t(model_end - model);
    auto *copy = static_cast<uint8_t *>(heap_caps_aligned_alloc(
        16, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (copy) {
        memcpy(copy, model, size);
        passed = benchmark("internal64k_model_psram", MALLOC_CAP_INTERNAL, copy) && passed;
        passed = benchmark("psram64k_model_psram", MALLOC_CAP_SPIRAM, copy) && passed;
        free(copy);
    } else {
        passed = false;
        printf("PROBE_HEAD {\"name\":\"psram_model\",\"allocated\":false}\n");
    }
    printf("P4_WAKE_PROBE_DONE {\"completed\":true,\"numeric_passed\":%s}\n",
        passed ? "true" : "false");
    fflush(stdout);
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}
