#pragma once
#include "portable/explicit_cache_bytes_v1.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <new>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace panel_wake_parallel {
constexpr size_t arena_size = 64 * 1024, variable_size = 16 * 1024;
constexpr char tag[] = "panel.wake.parallel";
inline nyx::Int8Bytes bytes(TfLiteTensor* tensor) {
    if (!tensor || tensor->type != kTfLiteInt8) return {nullptr, 0, 0, 0};
    return {tensor->data.int8, tensor->bytes, tensor->params.scale, tensor->params.zero_point};
}

inline bool shape(TfLiteTensor* tensor, std::initializer_list<int> dimensions) {
    if (!tensor || !tensor->dims || tensor->dims->size != int(dimensions.size())) return false;
    int index = 0;
    for (int dimension : dimensions) if (tensor->dims->data[index++] != dimension) return false;
    return true;
}

// Each model owns its arena and state. Explicit input caches must survive
// output-arena reuse; never feed an output view directly into the next Invoke.
struct Model {
    uint8_t* arena = nullptr;
    uint8_t* variable_arena = nullptr;
    tflite::MicroMutableOpResolver<15> ops;
    tflite::MicroInterpreter* interpreter = nullptr;
    const tflite::Model* model = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* probability = nullptr;
    TfLiteTensor* cache_inputs[7]{};
    TfLiteTensor* cache_outputs[7]{};
    nyx::Int8Bytes retained[7]{};
    bool explicit_cache = false;
    const char* name = nullptr;

    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    ~Model() {
        delete interpreter;
        for (auto& cache : retained) heap_caps_free(cache.data);
        heap_caps_free(variable_arena);
        heap_caps_free(arena);
    }

    // Resolve logical signature names from the original FlatBuffer. Tensor
    // index order differs from cache order (cache6 is input0 in these exports).
    TfLiteTensor* named_tensor(const char* name, bool output) {
        auto* signatures = model->signature_defs();
        if (!signatures || signatures->size() != 1) return nullptr;
        auto* signature = signatures->Get(0);
        if (signature->subgraph_index() != 0) return nullptr;
        auto* mappings = output ? signature->outputs() : signature->inputs();
        if (!mappings) return nullptr;
        auto* graph = model->subgraphs()->Get(0);
        auto* indices = output ? graph->outputs() : graph->inputs();
        for (auto* mapping : *mappings) {
            if (!mapping->name() || std::strcmp(mapping->name()->c_str(), name)) continue;
            for (unsigned i = 0; i < indices->size(); ++i) {
                if (indices->Get(i) == int(mapping->tensor_index()))
                    return output ? interpreter->output(i) : interpreter->input(i);
            }
        }
        return nullptr;
    }

    bool initialize(const uint8_t* data, const char* model_name, bool caches) {
        name = model_name;
        explicit_cache = caches;
        arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, arena_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        if (!arena) return false;
        tflite::MicroResourceVariables* variables = nullptr;
        if (!explicit_cache) {
            variable_arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, variable_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
            if (!variable_arena) return false;
            auto* allocator = tflite::MicroAllocator::Create(variable_arena, variable_size);
            if (!allocator) return false;
            variables = tflite::MicroResourceVariables::Create(allocator, 20);
            if (!variables) return false;
        }
        if (ops.AddCallOnce() != kTfLiteOk || ops.AddVarHandle() != kTfLiteOk ||
            ops.AddReadVariable() != kTfLiteOk || ops.AddReshape() != kTfLiteOk ||
            ops.AddConcatenation() != kTfLiteOk || ops.AddStridedSlice() != kTfLiteOk ||
            ops.AddAssignVariable() != kTfLiteOk || ops.AddConv2D() != kTfLiteOk ||
            ops.AddDepthwiseConv2D() != kTfLiteOk || ops.AddSplitV() != kTfLiteOk ||
            ops.AddFullyConnected() != kTfLiteOk || ops.AddLogistic() != kTfLiteOk ||
            ops.AddQuantize() != kTfLiteOk || ops.AddExpandDims() != kTfLiteOk ||
            ops.AddAdd() != kTfLiteOk) return false;
        model = tflite::GetModel(data);
        if (model->version() != TFLITE_SCHEMA_VERSION || !model->subgraphs() || !model->subgraphs()->size()) return false;
        interpreter = new(std::nothrow)tflite::MicroInterpreter(model, ops, arena, arena_size, variables);
        if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk) return false;
        int expected_count = explicit_cache ? 8 : 1;
        if (interpreter->inputs_size() != expected_count || interpreter->outputs_size() != expected_count) return false;
        input = named_tensor(explicit_cache ? "chunk" : "input_audio", false);
        probability = named_tensor(explicit_cache ? "probability" : "output_0", true);
        if (!nyx::valid_int8_bytes(bytes(input)) || input->bytes != 120) return false;
        if (explicit_cache) {
            if (!shape(input, {1, 3, 1, 40}) || !shape(probability, {1, 1, 1, 1})) return false;
            auto output_bytes = bytes(probability);
            if (!nyx::valid_int8_bytes(output_bytes) || output_bytes.size != 1 ||
                output_bytes.scale != 1.f / 256 || output_bytes.zero != -128) return false;
            constexpr int times[] = {2, 4, 8, 16, 32, 40, 1};
            for (int i = 0; i < 7; ++i) {
                char key[20];
                std::snprintf(key, sizeof(key), "cache%d", i);
                cache_inputs[i] = named_tensor(key, false);
                std::snprintf(key, sizeof(key), "cache%d_out", i);
                cache_outputs[i] = named_tensor(key, true);
                int channels = i == 0 ? 40 : 64;
                if (!shape(cache_inputs[i], {1, times[i], 1, channels}) ||
                    !shape(cache_outputs[i], {1, times[i], 1, channels})) return false;
                auto in = bytes(cache_inputs[i]), out = bytes(cache_outputs[i]);
                if (!nyx::valid_int8_bytes(in) || !nyx::valid_int8_bytes(out) ||
                    in.size != size_t(times[i] * channels) || out.size != in.size ||
                    in.scale != out.scale || in.zero != out.zero) return false;
                retained[i] = in;
                retained[i].data = static_cast<int8_t*>(heap_caps_malloc(in.size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
                if (!nyx::reset_explicit_cache(retained[i])) return false;
            }
        } else if (!shape(input, {1, 3, 40}) || !shape(probability, {1, 1}) ||
                   probability->type != kTfLiteUInt8 || probability->bytes != 1 ||
                   probability->params.scale != 1.f / 256 || probability->params.zero_point != 0) {
            return false;
        }
        size_t cache_bytes = 0;
        for (const auto& cache : retained) cache_bytes += cache.size;
        ESP_LOGI(tag, "model=%s arena_used=%u arena_alloc=%u variables_alloc=%u retained=%u",
                 name, unsigned(interpreter->arena_used_bytes()), unsigned(arena_size),
                 unsigned(variable_arena ? variable_size : 0), unsigned(cache_bytes));
        return true;
    }

    bool invoke(const float* features, uint8_t& code) {
        for (size_t i = 0; i < 120; ++i) {
            float quantized = std::nearbyint(features[i] / input->params.scale) + input->params.zero_point;
            input->data.int8[i] = int8_t(std::clamp(quantized, -128.f, 127.f));
        }
        if (explicit_cache) {
            for (int i = 0; i < 7; ++i)
                if (!nyx::feedback_explicit_cache(bytes(cache_inputs[i]), retained[i])) return false;
        }
        if (interpreter->Invoke() != kTfLiteOk) return false;
        if (explicit_cache) {
            if (!nyx::signed_probability_code(bytes(probability), code)) return false;
            for (int i = 0; i < 7; ++i)
                if (!nyx::feedback_explicit_cache(retained[i], bytes(cache_outputs[i]))) return false;
        } else code = probability->data.uint8[0];
        return true;
    }
};

} // namespace panel_wake_parallel
