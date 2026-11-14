// Host lifecycle witness: real registration and installed TFLM Prepare/Eval
// helpers, with a small host tensor/allocator context instead of a P4 arena.
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "complete_fc.h"

extern "C" int64_t esp_timer_get_time() { return 0; }

namespace {
struct Dims { int size; int data[3]; };
struct Scales { int size; float data[2]; };

class HostContext final : public tflite::MicroContext {
 public:
    TfLiteTensor tensors[4]{};
    TfLiteEvalTensor eval[4]{};
    int temporary_count=0;
    std::vector<void *> allocations;
    size_t persistent_bytes=0;
    ~HostContext() override { for(auto *p:allocations)std::free(p); }
    void *AllocatePersistentBuffer(size_t bytes) override {
        void *p=std::calloc(1,bytes);
        if(p) { allocations.push_back(p); persistent_bytes+=bytes; }
        return p;
    }
    TfLiteStatus RequestScratchBufferInArena(size_t,int *) override { return kTfLiteError; }
    void *GetScratchBuffer(int) override { return nullptr; }
    TfLiteTensor *AllocateTempTfLiteTensor(int index) override {
        if(index<0 || index>=4)return nullptr;
        ++temporary_count;
        return tensors+index;
    }
    void DeallocateTempTfLiteTensor(TfLiteTensor *) override { --temporary_count; }
    uint8_t *AllocateTempBuffer(size_t,size_t) override { return nullptr; }
    void DeallocateTempBuffer(uint8_t *) override {}
    TfLiteEvalTensor *GetEvalTensor(int index) override {
        if(index<0 || index>=4)return nullptr;
        eval[index].data=tensors[index].data;
        eval[index].dims=tensors[index].dims;
        eval[index].type=tensors[index].type;
        return eval+index;
    }
    TfLiteStatus set_external_context(void *) override { return kTfLiteError; }
    void *external_context() override { return nullptr; }
    tflite::MicroGraph &graph() override { std::abort(); }
 private:
    TF_LITE_REMOVE_VIRTUAL_DELETE
};

void report(TfLiteContext *,const char *format,...) {
    va_list args;
    va_start(args,format);
    std::vfprintf(stderr,format,args);
    std::fputc('\n',stderr);
    va_end(args);
}
} // namespace

extern "C" {
struct CompleteFcHostConfig {
    int batches,depth,input_zero,filter_zero,output_zero,activation,variant;
    float input_scale,filter_scale,bias_scale,output_scale;
};
struct CompleteFcHostReceipt {
    int prepare_status,invoke_status,temporary_count,activation_min,activation_max;
    int input_zero,output_zero;
    double effective_scale;
    size_t persistent_bytes;
};

__attribute__((visibility("default"))) int complete_fc_registration_host_v1(
    const int8_t *input,const int8_t *weights,const int32_t *bias,int8_t *output,
    const CompleteFcHostConfig *config,CompleteFcHostReceipt *receipt) {
    HostContext host;
    TfLiteContext context{};
    context.impl_=&host;
    context.AllocatePersistentBuffer=tflite::MicroContextAllocatePersistentBuffer;
    context.GetEvalTensor=tflite::MicroContextGetEvalTensor;
    context.ReportError=report;
    Dims shapes[4]={{2,{config->batches,config->depth,0}},
                    {2,{1,config->depth,0}}, {1,{1,0,0}},
                    {2,{config->batches,1,0}}};
    for(int i=0;i<4;++i) {
        host.tensors[i].type=i==2?kTfLiteInt32:kTfLiteInt8;
        host.tensors[i].dims=reinterpret_cast<TfLiteIntArray *>(shapes+i);
    }
    host.tensors[0].data.int8=const_cast<int8_t *>(input);
    host.tensors[0].params={config->input_scale,config->input_zero};
    host.tensors[1].data.int8=const_cast<int8_t *>(weights);
    host.tensors[1].params={config->filter_scale,config->filter_zero};
    Scales filter_scales={1,{config->filter_scale,config->filter_scale}};
    Dims filter_zeros={1,{config->filter_zero,config->filter_zero,0}};
    TfLiteAffineQuantization filter_affine{};
    filter_affine.scale=reinterpret_cast<TfLiteFloatArray *>(&filter_scales);
    filter_affine.zero_point=reinterpret_cast<TfLiteIntArray *>(&filter_zeros);
    filter_affine.quantized_dimension=0;
    host.tensors[1].quantization={kTfLiteAffineQuantization,&filter_affine};
    host.tensors[2].data.i32=const_cast<int32_t *>(bias);
    host.tensors[2].params={config->bias_scale,0};
    host.tensors[3].data.int8=output;
    host.tensors[3].params={config->output_scale,config->output_zero};
    // Guard probes never invoke unsupported metadata. No injected standard
    // Prepare failure is used: all statuses come from real source functions.
    if(config->variant==1)shapes[1].data[0]=2;  // multi-output, unsupported
    if(config->variant==2)shapes[2].data[0]=2;  // non-scalar bias
    if(config->variant==3)host.tensors[3].type=kTfLiteUInt8;
    if(config->variant==5) {
        shapes[1].data[0]=2;
        filter_scales.size=2;
        filter_zeros.size=2;
    }
    Dims inputs={3,{0,1,bias?2:kTfLiteOptionalTensor}};
    Dims outputs={1,{3,0,0}};
    TfLiteFullyConnectedParams params{};
    params.activation=static_cast<TfLiteFusedActivation>(config->activation);
    TfLiteNode node{};
    node.inputs=reinterpret_cast<TfLiteIntArray *>(&inputs);
    node.outputs=reinterpret_cast<TfLiteIntArray *>(&outputs);
    node.builtin_data=&params;
    auto registration=nyx_complete_fc::registration();
    node.user_data=registration.init(&context,nullptr,0);
    if(!node.user_data)return -1;
    receipt->prepare_status=registration.prepare(&context,&node);
    receipt->invoke_status=-1;
    if(receipt->prepare_status==kTfLiteOk) {
        const auto &data=*static_cast<nyx_complete_fc::Data *>(node.user_data);
        receipt->activation_min=data.original.output_activation_min;
        receipt->activation_max=data.original.output_activation_max;
        receipt->input_zero=data.original.input_zero_point;
        receipt->output_zero=data.original.output_zero_point;
        receipt->effective_scale=data.effective_scale;
        if(config->variant==4)shapes[0].data[1]+=1; // changed shape at Invoke
        receipt->invoke_status=registration.invoke(&context,&node);
    }
    receipt->temporary_count=host.temporary_count;
    receipt->persistent_bytes=host.persistent_bytes;
    return 0;
}
}
