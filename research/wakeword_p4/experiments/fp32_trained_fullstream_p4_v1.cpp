// Original trained-model numeric/resource probe. No audio, network or NVS.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "fixtures.h"

static_assert(sizeof(float) == 4, "The frozen inputs and scores use float32");
constexpr size_t kMainBytes = 256 * 1024, kVariableBytes = 64 * 1024;
constexpr int64_t kDeadlineUs = 30000;
constexpr float kTolerance = 1e-4f;

static bool inner(unsigned case_id, unsigned repeat, uint8_t *arena, uint8_t *variable) {
    const auto &item = kCases[case_id];
    const auto &stream = kStreams[item.stream];
    const int64_t setup_start = esp_timer_get_time();
    auto *allocator = tflite::MicroAllocator::Create(variable, kVariableBytes);
    if (!allocator) return false;
    auto *variables = tflite::MicroResourceVariables::Create(allocator, 20);
    if (!variables) return false;
    tflite::MicroMutableOpResolver<12> ops;
    const TfLiteStatus registrations[] = {
        ops.AddCallOnce(), ops.AddVarHandle(), ops.AddReadVariable(), ops.AddReshape(),
        ops.AddConcatenation(), ops.AddStridedSlice(), ops.AddAssignVariable(),
        ops.AddConv2D(), ops.AddDepthwiseConv2D(), ops.AddFullyConnected(),
        ops.AddLogistic(), ops.AddExpandDims()
    };
    for (auto status : registrations) if (status != kTfLiteOk) return false;
    const auto *model = tflite::GetModel(kModels[item.model].data);
    if (model->version() != TFLITE_SCHEMA_VERSION) return false;
    tflite::MicroInterpreter interpreter(model, ops, arena, kMainBytes, variables);
    if (interpreter.inputs_size() != 1 || interpreter.outputs_size() != 1 ||
        interpreter.AllocateTensors() != kTfLiteOk) return false;
    const auto setup_us = esp_timer_get_time() - setup_start;
    auto *input = interpreter.input(0); auto *output = interpreter.output(0);
    if (!input || !output || input->type != kTfLiteFloat32 || input->bytes != 480 ||
        input->dims->size != 3 || input->dims->data[0] != 1 ||
        input->dims->data[1] != 3 || input->dims->data[2] != 40 ||
        output->type != kTfLiteFloat32 || output->bytes != 4 ||
        output->dims->size != 2 || output->dims->data[0] != 1 || output->dims->data[1] != 1) return false;
    float max_error = 0;
    int64_t max_us = 0;
    bool passed = true;
    for (size_t step = 0; step < stream.steps; ++step) {
        std::memcpy(input->data.f, stream.input + step * 480, 480);
        const auto start = esp_timer_get_time();
        const auto status = interpreter.Invoke();
        const auto elapsed = esp_timer_get_time() - start;
        if (status != kTfLiteOk) return false;
        float expected;
        std::memcpy(&expected, item.expected_tf + step * 4, 4);
        const auto actual = output->data.f[0];
        uint32_t bits;
        std::memcpy(&bits, &actual, 4);
        if (!std::isfinite(actual) || actual < 0 || actual > 1 || elapsed <= 0) passed = false;
        max_error = std::max(max_error, std::fabs(actual - expected));
        max_us = std::max(max_us, elapsed);
        // Emit all output bytes and all Invoke durations, including cold calls.
        // UART and the scheduler yield are outside the measured Invoke region.
        printf("P4_TRAINED_STEP {\"case\":%u,\"repeat\":%u,\"step\":%u,\"bits\":\"%08x\",\"us\":%lld}\n",
               case_id, repeat, unsigned(step), unsigned(bits), (long long)elapsed);
        if ((step + 1) % 16 == 0) vTaskDelay(1);
    }
    passed = passed && max_error <= kTolerance && max_us <= kDeadlineUs;
    printf("P4_TRAINED_CASE {\"case\":%u,\"repeat\":%u,\"steps\":%u,\"passed\":%s,\"max_abs_tf\":%.9g,\"max_us\":%lld,\"setup_us\":%lld,\"arena_used\":%u,\"variable_used\":%u}\n",
           case_id, repeat, unsigned(stream.steps), passed ? "true" : "false", double(max_error),
           (long long)max_us, (long long)setup_us, unsigned(interpreter.arena_used_bytes()),
           unsigned(allocator->used_bytes()));
    return passed;
}

static bool run(unsigned case_id, unsigned repeat) {
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, kMainBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    auto *variable = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, kVariableBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    bool passed = false;
    if (arena && variable) {
        // Match the value-initialized arenas in the admitted native host probe.
        std::memset(arena, 0, kMainBytes);
        std::memset(variable, 0, kVariableBytes);
        passed = inner(case_id, repeat, arena, variable);
    }
    if (!passed) printf("P4_TRAINED_ERROR {\"case\":%u,\"repeat\":%u}\n", case_id, repeat);
    free(arena); free(variable);
    return passed;
}

extern "C" void app_main() {
    printf("P4_TRAINED_BEGIN {\"cases\":%u,\"repeats\":2,\"configured_cpu_mhz\":%u,\"arena_bytes\":%u,\"variable_bytes\":%u}\n",
           unsigned(kCaseCount), unsigned(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ), unsigned(kMainBytes), unsigned(kVariableBytes));
    bool passed = true;
    for (unsigned case_id = 0; case_id < kCaseCount; ++case_id)
        for (unsigned repeat = 0; repeat < 2; ++repeat)
            passed = run(case_id, repeat) && passed;
    printf("P4_TRAINED_DONE {\"passed\":%s,\"cases\":%u,\"repeats\":2,\"stack_free\":%u}\n",
           passed ? "true" : "false", unsigned(kCaseCount), unsigned(uxTaskGetStackHighWaterMark(nullptr)));
}
