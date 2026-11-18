#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "../../portable/c161_binary40_events_v1.h"

extern const uint8_t model_bytes[] asm("_binary_4090_c161_midlevel_single_v1_tflite_start");
extern const uint8_t fixture[] asm("_binary_midlevel_bin_start");
extern const uint8_t fixture_end[] asm("_binary_midlevel_bin_end");

namespace {
constexpr unsigned stride = 124;
#ifdef NYX_TRACE_TENSORS
constexpr bool trace_tensors = true;
#else
constexpr bool trace_tensors = false;
#endif
tflite::MicroMutableOpResolver<16> resolver;

struct Lane {
    alignas(16) uint8_t tensors[65536];
    alignas(16) uint8_t variables[16384];
    std::optional<tflite::MicroInterpreter> model;
    tflite::MicroAllocator* allocator{};
    TfLiteTensor* input{};
    TfLiteTensor* output{};
    unsigned calls{};

    bool reset() {
        model.reset();
        std::memset(tensors, 0, sizeof(tensors));
        std::memset(variables, 0, sizeof(variables));
        allocator = tflite::MicroAllocator::Create(variables, sizeof(variables));
        if (!allocator) return false;
        auto* state = tflite::MicroResourceVariables::Create(allocator, 24);
        if (!state) return false;
        model.emplace(tflite::GetModel(model_bytes), resolver, tensors, sizeof(tensors), state, nullptr, trace_tensors);
        if (model->AllocateTensors() != kTfLiteOk) return false;
        input = model->input(0);
        output = model->output(0);
        return input && input->type == kTfLiteInt8 && input->bytes == 120 &&
               output && output->type == kTfLiteUInt8 && output->bytes == 1;
    }

    void trace(unsigned scene, unsigned step, unsigned lane) {
        for (int index = 38; index <= 74; ++index) {
            auto* t = model->GetTensor(index);
            if (t->type != kTfLiteInt8 && t->type != kTfLiteUInt8) continue;
            int elements = 1;
            for (int d = 0; d < t->dims->size; ++d) elements *= t->dims->data[d];
            uint32_t hash = 2166136261u;
            for (int j = 0; j < elements; ++j) hash = (hash ^ t->data.uint8[j]) * 16777619u;
            std::printf("MIDLEVEL_TENSOR source=%u step=%u lane=%u tensor=%d bytes=%d fnv=%08lx first=%u\n",
                        scene, step, lane, index, elements, (unsigned long)hash, t->data.uint8[0]);
        }
    }

    bool invoke(const uint8_t* frames) {
        std::memcpy(input->data.int8, frames, 120);
        ++calls;
        return model->Invoke() == kTfLiteOk;
    }
};

Lane normal, expanded;
unsigned blocks, errors, event_errors;
int64_t max_us;
unsigned over_budget;

uint32_t word(const uint8_t* ptr) {
    uint32_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

bool replay(unsigned scene, unsigned length, const uint8_t* data) {
    if (!normal.reset() || !expanded.reset()) return false;
    uint8_t pending[120];
    unsigned used = 0, events[2]{};
    nyx::Detector detectors[2];
    for (unsigned step = 0; step < length; ++step) {
        const auto* row = data + step * stride;
        const auto start = esp_timer_get_time();
        if (!normal.invoke(row)) return false;
        for (unsigned j = 0; j < 3; ++j) {
            const unsigned copies = 1 + ((step * 3 + j) % 2);
            for (unsigned k = 0; k < copies; ++k) {
                std::memcpy(pending + 40 * used, row + 40 * j, 40);
                if (++used == 3) {
                    if (!expanded.invoke(pending)) return false;
                    used = 0;
                }
            }
        }
        const uint8_t codes[] = {normal.output->data.uint8[0], expanded.output->data.uint8[0]};
        for (unsigned i = 0; i < 2; ++i) {
            const bool fired = detectors[i].update(step + 1, codes[i]);
            if (codes[i] != row[120 + i]) {
                ++errors;
                if (trace_tensors) (i == 0 ? normal : expanded).trace(scene, step + 1, i);
                std::printf("MIDLEVEL_DIFF source=%u step=%u lane=%u expected=%u actual=%u\n",
                            scene, step + 1, i, row[120 + i], codes[i]);
            }
            event_errors += fired != bool(row[122 + i]);
            events[i] += fired;
        }
        const auto elapsed = esp_timer_get_time() - start;
        max_us = std::max(max_us, elapsed);
        over_budget += elapsed >= 30000;
    }
    blocks += length;
    std::printf("MIDLEVEL_STREAM source=%u blocks=%u proposals=%u,%u\n", scene, length, events[0], events[1]);
    return true;
}
}

static void run() {
    resolver.AddCallOnce(); resolver.AddVarHandle(); resolver.AddReadVariable(); resolver.AddReshape();
    resolver.AddConcatenation(); resolver.AddStridedSlice(); resolver.AddAssignVariable(); resolver.AddConv2D();
    resolver.AddDepthwiseConv2D(); resolver.AddSplitV(); resolver.AddFullyConnected(); resolver.AddLogistic();
    resolver.AddQuantize(); resolver.AddExpandDims();
    const auto* cursor = fixture;
    if (fixture_end - cursor < 4 || word(cursor) != 12) { std::puts("MIDLEVEL_FAIL header"); return; }
    cursor += 4;
    for (unsigned i = 0; i < 12; ++i) {
        if (fixture_end - cursor < 8) { std::puts("MIDLEVEL_FAIL stream"); return; }
        const unsigned scene = word(cursor), length = word(cursor + 4);
        cursor += 8;
        if (!length || length > (fixture_end - cursor) / stride || !replay(scene, length, cursor)) {
            std::puts("MIDLEVEL_FAIL replay"); return;
        }
        cursor += length * stride;
        vTaskDelay(1);
    }
    const bool pass = cursor == fixture_end && !errors && !event_errors;
    std::printf("MIDLEVEL_%s blocks=%u calls=%u,%u score_errors=%u event_errors=%u tensors=%u,%u variables=%u,%u max_us=%lld over30ms=%u min_internal=%u\n",
                pass ? "PASS" : "FAIL", blocks, normal.calls, expanded.calls, errors, event_errors,
                unsigned(normal.model->arena_used_bytes()), unsigned(expanded.model->arena_used_bytes()),
                unsigned(normal.allocator->used_bytes()), unsigned(expanded.allocator->used_bytes()),
                (long long)max_us, over_budget, unsigned(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
}

extern "C" void app_main() {
    run();
    std::puts("MIDLEVEL_DONE");
}
