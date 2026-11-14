// Opt-in scalar FC compatibility for the complete teacher60 research runtime.
// Dependency kernels and legacy/embedding registrations remain independent.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include "complete_fc_math.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace nyx_complete_fc {
struct Data {
    tflite::OpDataFullyConnected original;
    double effective_scale;
};
static_assert(std::is_standard_layout_v<Data> && offsetof(Data,original)==0);

inline void *initialize(TfLiteContext *context,const char *,size_t) {
    void *storage=context->AllocatePersistentBuffer(context,sizeof(Data));
    return storage?new(storage)Data{}:nullptr;
}

inline TfLiteStatus prepare(TfLiteContext *context,TfLiteNode *node) {
    TF_LITE_ENSURE(context,node->user_data);
    auto *micro=tflite::GetMicroContext(context);
    auto *input=micro->AllocateTempInputTensor(node,0);
    auto *filter=micro->AllocateTempInputTensor(node,1);
    auto *bias=micro->AllocateTempInputTensor(node,2);
    auto *output=micro->AllocateTempOutputTensor(node,0);
    auto &data=*static_cast<Data *>(node->user_data);
    bool supported=input && filter && output;
    if(supported) {
        bool scalar=true;
        if(filter->quantization.type==kTfLiteAffineQuantization) {
            const auto *affine=static_cast<const TfLiteAffineQuantization *>(
                filter->quantization.params);
            scalar=affine && affine->scale && affine->scale->size==1;
        }
        supported=input->type==kTfLiteInt8 && filter->type==kTfLiteInt8 &&
            output->type==kTfLiteInt8 && filter->dims && filter->dims->size==2 &&
            filter->dims->data[0]==1 && filter->dims->data[1]>0 &&
            filter->dims->data[1]<=4096 && scalar &&
            filter->params.zero_point==0 && input->params.zero_point>=-128 &&
            input->params.zero_point<=127 && output->params.zero_point>=-128 &&
            output->params.zero_point<=127 && std::isfinite(input->params.scale) &&
            input->params.scale>0 && std::isfinite(filter->params.scale) &&
            filter->params.scale>0 && std::isfinite(output->params.scale) &&
            output->params.scale>0;
        if(supported) {
            data.effective_scale=double(input->params.scale)*
                double(filter->params.scale)/double(output->params.scale);
            supported=std::isfinite(data.effective_scale) && data.effective_scale>0;
            const int64_t worst=int64_t(filter->dims->data[1])*255*128;
            int64_t bias_value=0;
            if(bias) {
                supported=supported && bias->type==kTfLiteInt32 && bias->dims &&
                    bias->dims->size==1 && bias->dims->data[0]==1 && bias->data.i32;
                if(supported)bias_value=bias->data.i32[0];
            }
            supported=supported && std::abs(bias_value)+worst<=std::numeric_limits<int32_t>::max();
        }
    }
    if(input)micro->DeallocateTempTfLiteTensor(input);
    if(filter)micro->DeallocateTempTfLiteTensor(filter);
    if(bias)micro->DeallocateTempTfLiteTensor(bias);
    if(output)micro->DeallocateTempTfLiteTensor(output);
    TF_LITE_ENSURE(context,supported);
    // Validate supported metadata first: the installed standard Prepare returns
    // early without releasing its temporary tensors on a type mismatch.
    // It then supplies activation bounds and zero-points in the first member.
    TF_LITE_ENSURE_OK(context,tflite::Register_FULLY_CONNECTED().prepare(context,node));
    TF_LITE_ENSURE(context,!data.original.is_per_channel &&
        data.original.output_activation_min>=-128 &&
        data.original.output_activation_max<=127 &&
        data.original.output_activation_min<=data.original.output_activation_max);
    return kTfLiteOk;
}

inline TfLiteStatus evaluate(TfLiteContext *context,TfLiteNode *node) {
    const auto &data=*static_cast<const Data *>(node->user_data);
    const auto *input=tflite::micro::GetEvalInput(context,node,0);
    const auto *filter=tflite::micro::GetEvalInput(context,node,1);
    const auto *bias=tflite::micro::GetEvalInput(context,node,2);
    auto *output=tflite::micro::GetEvalOutput(context,node,0);
    const int depth=filter->dims->data[1];
    const int batches=tflite::micro::GetTensorShape(output).FlatSize();
    TF_LITE_ENSURE(context,batches>0 &&
        tflite::micro::GetTensorShape(input).FlatSize()==batches*depth);
    scalar_fc(input->data.int8,filter->data.int8,
        tflite::micro::GetOptionalTensorData<int32_t>(bias),output->data.int8,
        batches,depth,data.original.input_zero_point,data.original.output_zero_point,
        data.effective_scale,data.original.output_activation_min,data.original.output_activation_max);
    return kTfLiteOk;
}

inline TFLMRegistration registration() {
    auto op=tflite::Register_FULLY_CONNECTED();
    op.init=initialize;
    op.prepare=prepare;
    op.invoke=evaluate;
    return op;
}
} // namespace nyx_complete_fc
