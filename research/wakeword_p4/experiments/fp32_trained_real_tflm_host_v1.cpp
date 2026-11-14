// Trained-model runner: original float kernels plus installed ExpandDims; host only.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"

static_assert(sizeof(float) == 4, "The fixed vectors use float32");

struct alignas(16) Arenas {
    std::array<uint8_t, 256 * 1024> main;
    std::array<uint8_t, 64 * 1024> variable;
};

static std::vector<char> read(const char *path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const auto model = read(argv[1]);
    const auto inputs = read(argv[2]);
    if (model.empty() || inputs.size() != 4 * 256 * 120 * sizeof(float)) return 3;
    std::ofstream output(argv[3], std::ios::binary);
    if (!output) return 4;
    for (unsigned condition = 0; condition < 4; ++condition) {
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            auto arena = std::make_unique<Arenas>();
            auto *allocator = tflite::MicroAllocator::Create(arena->variable.data(), arena->variable.size());
            if (!allocator) return 11;
            auto *variables = tflite::MicroResourceVariables::Create(allocator, 20);
            if (!variables) return 12;
            tflite::MicroMutableOpResolver<12> ops;
            const TfLiteStatus registrations[] = {
                ops.AddCallOnce(), ops.AddVarHandle(), ops.AddReadVariable(), ops.AddReshape(),
                ops.AddConcatenation(), ops.AddStridedSlice(), ops.AddAssignVariable(),
                ops.AddConv2D(), ops.AddDepthwiseConv2D(), ops.AddFullyConnected(),
                ops.AddLogistic(), ops.AddExpandDims()
            };
            for (auto status : registrations) if (status != kTfLiteOk) return 10;
            tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), ops, arena->main.data(), arena->main.size(), variables);
            if (interpreter.inputs_size() != 1 || interpreter.outputs_size() != 1) return 13;
            if (interpreter.AllocateTensors() != kTfLiteOk) return 5;
            auto *in = interpreter.input(0); auto *out = interpreter.output(0);
            if (!in || !out || in->dims->size != 3 || in->dims->data[0] != 1 ||
                in->dims->data[1] != 3 || in->dims->data[2] != 40 || out->dims->size != 2 ||
                out->dims->data[0] != 1 || out->dims->data[1] != 1) return 14;
            if (in->type != kTfLiteFloat32 || in->bytes != 480 || out->type != kTfLiteFloat32 || out->bytes != 4) return 6;
            for (unsigned step = 0; step < 256; ++step) {
                const auto offset = (condition * 256 + step) * 120 * sizeof(float);
                std::memcpy(in->data.f, inputs.data() + offset, 480);
                if (interpreter.Invoke() != kTfLiteOk || !std::isfinite(out->data.f[0])) return 7;
                output.write(reinterpret_cast<const char *>(out->data.f), sizeof(float));
                if (!output) return 8;
            }
            printf("HOST_TFLM_CASE {\"condition\":%u,\"repeat\":%u,\"steps\":256,\"arena_used\":%zu,\"variable_used\":%zu}\n",
                   condition, repeat, interpreter.arena_used_bytes(), allocator->used_bytes());
        }
    }
    output.close();
    if (!output) return 9;
    puts("HOST_TFLM_DONE");
    return 0;
}
