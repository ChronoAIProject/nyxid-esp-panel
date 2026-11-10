// Optional numerical diagnosis. Trace output adds UART time; do not use its
// maximum invocation time as a performance measurement.
#pragma once
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
namespace nyx_head_probe {
inline bool tracing = false;
inline TfLiteStatus (*original_eval[256])(TfLiteContext *, TfLiteNode *){};
template <tflite::BuiltinOperator Code>
TfLiteStatus trace_eval(TfLiteContext *context, TfLiteNode *node) {
    auto status = original_eval[Code] ? original_eval[Code](context, node) : kTfLiteOk;
    if (tracing && status == kTfLiteOk) {
        for (int n = 0; n < node->outputs->size; ++n) {
            const auto *out = tflite::micro::GetEvalOutput(context, node, n);
            int bytes = tflite::micro::GetTensorShape(out).FlatSize();
            if (out->type == kTfLiteFloat32 || out->type == kTfLiteInt32) bytes *= 4;
            printf("HEAD_TRACE {\"opcode\":%d,\"tensor\":%d,\"type\":%d,\"hex\":\"",
                int(Code), node->outputs->data[n], int(out->type));
            const auto *raw = reinterpret_cast<const uint8_t *>(out->data.raw);
            for (int k = 0; k < bytes; ++k) printf("%02x", unsigned(raw[k]));
            printf("\"}\n");
        }
    }
    return status;
}
template <tflite::BuiltinOperator Code, unsigned Count>
void trace_registration(tflite::MicroMutableOpResolver<Count> &ops) {
    auto *entry = const_cast<TFLMRegistration *>(ops.FindOp(Code));
    original_eval[Code] = entry->invoke;
    entry->invoke = trace_eval<Code>;
}
template <unsigned Count>
void trace_registrations(tflite::MicroMutableOpResolver<Count> &ops) {
    trace_registration<tflite::BuiltinOperator_RESHAPE>(ops);
    trace_registration<tflite::BuiltinOperator_FULLY_CONNECTED>(ops);
    trace_registration<tflite::BuiltinOperator_MEAN>(ops);
    trace_registration<tflite::BuiltinOperator_DEQUANTIZE>(ops);
    trace_registration<tflite::BuiltinOperator_NEG>(ops);
    trace_registration<tflite::BuiltinOperator_QUANTIZE>(ops);
    trace_registration<tflite::BuiltinOperator_SQUARED_DIFFERENCE>(ops);
    trace_registration<tflite::BuiltinOperator_ADD>(ops);
    trace_registration<tflite::BuiltinOperator_RSQRT>(ops);
    trace_registration<tflite::BuiltinOperator_MUL>(ops);
    trace_registration<tflite::BuiltinOperator_LOGISTIC>(ops);
}
} // namespace nyx_head_probe
