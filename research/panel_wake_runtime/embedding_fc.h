// Local int8 FC compatibility for esp-tflite-micro 1.3.3~1 / ESP-NN 1.1.2.
// Preparation supports per-channel scales, but ESP-NN evaluation ignores them.
// LiteRT BUILTIN_REF also uses single-rounding requantization, while the pinned
// MCU reference path double-rounds. Keep dependency sources unchanged and make
// these numerical choices explicit in this experimental registration only.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace nyx_head_probe {
inline TfLiteStatus reference_fc_prepare(TfLiteContext *context, TfLiteNode *node) {
    TF_LITE_ENSURE_OK(context, tflite::Register_FULLY_CONNECTED().prepare(context, node));
    auto *micro = tflite::GetMicroContext(context);
    auto *input = micro->AllocateTempInputTensor(node, 0);
    auto *filter = micro->AllocateTempInputTensor(node, 1);
    auto *bias = micro->AllocateTempInputTensor(node, 2);
    auto *output = micro->AllocateTempOutputTensor(node, 0);
    auto &data = *static_cast<tflite::OpDataFullyConnected *>(node->user_data);
    bool supported = true;
    if (input->type == kTfLiteInt8 && filter->type == kTfLiteInt8) {
        supported = filter->dims->size == 2 && filter->dims->data[1] > 0 &&
            filter->dims->data[1] <= 4096 && filter->params.zero_point == 0;
        const auto *quant = static_cast<const TfLiteAffineQuantization *>(filter->quantization.params);
        supported = supported && quant && (!data.is_per_channel || quant->quantized_dimension == 0);
        if (supported) {
            const int64_t worst = int64_t(filter->dims->data[1]) * 255 * 128;
            if (bias) for (int n = 0; n < filter->dims->data[0]; ++n) {
                int64_t value = bias->data.i32[n];
                supported = supported && std::abs(value) + worst <= std::numeric_limits<int32_t>::max();
            }
            if (!data.is_per_channel) {
                // Use the same double-precision scale product as the host's FC.
                double scale = double(input->params.scale) * double(filter->params.scale) / double(output->params.scale);
                tflite::QuantizeMultiplier(scale, &data.output_multiplier, &data.output_shift);
            }
        }
    }
    micro->DeallocateTempTfLiteTensor(input);
    micro->DeallocateTempTfLiteTensor(filter);
    if (bias) micro->DeallocateTempTfLiteTensor(bias);
    micro->DeallocateTempTfLiteTensor(output);
    TF_LITE_ENSURE(context, supported);
    return kTfLiteOk;
}

inline TfLiteStatus reference_fc_eval(TfLiteContext *context, TfLiteNode *node) {
    const auto &data = *static_cast<const tflite::OpDataFullyConnected *>(node->user_data);
    const auto *input = tflite::micro::GetEvalInput(context, node, 0);
    const auto *filter = tflite::micro::GetEvalInput(context, node, 1);
    const auto *bias = tflite::micro::GetEvalInput(context, node, 2);
    auto *output = tflite::micro::GetEvalOutput(context, node, 0);
    if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8)
        return tflite::Register_FULLY_CONNECTED().invoke(context, node);
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);
    const int depth = filter->dims->data[1], channels = filter->dims->data[0];
    const int values = tflite::micro::GetTensorShape(output).FlatSize();
    TF_LITE_ENSURE(context, channels > 0 && values % channels == 0);
    const int batches = values / channels;
    TF_LITE_ENSURE_EQ(context, tflite::micro::GetTensorShape(input).FlatSize(), batches * depth);
    const auto *bias_data = tflite::micro::GetOptionalTensorData<int32_t>(bias);
    for (int b = 0; b < batches; ++b) for (int c = 0; c < channels; ++c) {
        int32_t acc = bias_data ? bias_data[c] : 0;
        for (int k = 0; k < depth; ++k)
            acc += int32_t(filter->data.int8[c * depth + k]) *
                (int32_t(input->data.int8[b * depth + k]) - data.input_zero_point);
        const int32_t multiplier = data.is_per_channel ? data.per_channel_output_multiplier[c] : data.output_multiplier;
        const int shift = data.is_per_channel ? data.per_channel_output_shift[c] : data.output_shift;
        TF_LITE_ENSURE(context, shift >= -31 && shift <= 30 && multiplier >= 0);
        const int total_shift = 31 - shift;
        const int64_t rounded = int64_t(acc) * multiplier + (int64_t(1) << (total_shift - 1));
        int64_t value = (rounded >> total_shift) + data.output_zero_point;
        value = std::clamp(value, int64_t(data.output_activation_min), int64_t(data.output_activation_max));
        output->data.int8[b * channels + c] = int8_t(value);
    }
    return kTfLiteOk;
}

inline TFLMRegistration reference_fc_registration() {
    auto registration = tflite::Register_FULLY_CONNECTED();
    registration.prepare = reference_fc_prepare;
    registration.invoke = reference_fc_eval;
    return registration;
}
} // namespace nyx_head_probe
